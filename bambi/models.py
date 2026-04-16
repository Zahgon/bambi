# pylint: disable=no-name-in-module
# pylint: disable=too-many-lines
# pylint: disable=too-many-positional-arguments
import logging
import warnings

from copy import deepcopy
from importlib.metadata import version

import formulae as fm
import pymc as pm
import pandas as pd

from arviz_plots import plot_dist
from arviz_stats import residual_r2

from bambi.backend import PyMCModel
from bambi.defaults import get_builtin_family
from bambi.model_components import ConstantComponent, DistributionalComponent, ResponseComponent
from bambi.families import Family, univariate
from bambi.formula import Formula, check_ordinal_formula
from bambi.priors import Prior, PriorScaler
from bambi.transformations import transformations_namespace
from bambi.utils import (
    clean_formula_lhs,
    get_aliased_name,
    indentify,
    listify,
    remove_common_intercept,
    wrapify,
)

_log = logging.getLogger("bambi")

ORDINAL_FAMILIES = (univariate.Cumulative, univariate.StoppingRatio)

__version__ = version("bambi")


class Model:
    """Specification of model class

    Parameters
    ----------
    formula : str or Formula
        A model description written using the formula syntax from the `formulae` library.
    data : pd.DataFrame
        A pandas dataframe containing the data on which the model will be fit, with column
        names matching variables defined in the formula.
    family : str or bambi.Family, optional
        A specification of the model family (analogous to the family object in R). Either
        a string, or an instance of class [](`bambi.Family`). If a string is passed, a
        family with the corresponding name must be defined in the defaults loaded at `Model`
        initialization. Valid pre-defined families are `"bernoulli"`, `"beta"`,
        `"binomial"`, `"categorical"`, `"gamma"`, `"gaussian"`, `"negativebinomial"`,
        `"poisson"`, `"t"`, and `"wald"`. Defaults to `"gaussian"`.
    priors : dict, optional
        Optional specification of priors for one or more terms. A dictionary where the keys are
        the names of terms in the model, "common," or "group_specific" and the values are
        instances of class `Prior`. If priors are unset, use automatic priors inspired by
        the R rstanarm library.
    link : str or dict of str to str, optional
        The name of the link function to use. Valid names are `"cloglog"`, `"identity"`,
        `"inverse_squared"`, `"inverse"`, `"log"`, `"logit"`, `"probit"`, and
        `"softmax"`. Not all the link functions can be used with all the families.
        If a dictionary, keys are the names of the target parameters and the values are the names
        of the link functions.
    categorical : str or list of str, optional
        The names of any variables to treat as categorical. Can be either a single variable
        name, or a list of names. If categorical is `None`, the data type of the columns in
        the `data` will be used to infer handling. In cases where numeric columns are
        to be treated as categorical (e.g., group specific factors coded as numerical IDs),
        explicitly passing variable names via this argument is recommended.
    potentials : A list of 2-tuples, optional
        Optional specification of potentials. A potential is an arbitrary expression added to the
        likelihood, this is generally useful to add constrains to models, that are difficult to
        express otherwise. The first term of a 2-tuple is the name of a variable in the model, the
        second a lambda function expressing the desired constraint.
        If a constraint involves n variables, you can pass n 2-tuples or pass a tuple which first
        element is an n-tuple and second element is a lambda function with n arguments. The number
        and order of the lambda function has to match the number and order of the variable names.
    dropna : bool, optional
        When `True`, rows with any missing values in either the predictors or outcome are
        automatically dropped from t, optionalhe dataset in a listwise manner.
    auto_scale : bool
        If `True` (default), priors are automatically rescaled to the data
        (to be weakly informative) any time default priors are used. Note that any priors
        explicitly set by the user will always take precedence over default priors.
    noncentered : bool, optional
        If `True` (default), uses a non-centered parameterization for normal hyperpriors on
        grouped parameters. If `False`, naive (centered) parameterization is used.
    center_predictors : bool, optional
        If `True` (default), and if there is an intercept in the common terms, the data is
        centered by subtracting the mean. The centering is undone after sampling to provide
        the actual intercept in all distributional components that have an intercept. Note
        that this changes the interpretation of the prior on the intercept because it refers
        to the intercept of the centered data.
    extra_namespace : dict, optional
        Additional user supplied variables with transformations or data to include in the
        environment where the formula is evaluated. Defaults to `None`.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        formula,
        data,
        family="gaussian",
        priors=None,
        link=None,
        categorical=None,
        potentials=None,
        dropna=False,
        auto_scale=True,
        noncentered=True,
        center_predictors=True,
        extra_namespace=None,
    ):
        # attributes that are set later
        self.components = {}  # Constant and Distributional components
        self.built = False  # build()

        # build() will loop over this, calling _set_priors()
        self._added_priors = {}

        self.family = None  # _add_response()
        self.backend = None  # _set_backend()

        self.auto_scale = auto_scale
        self.dropna = dropna
        self.formula = formula
        self.noncentered = noncentered
        self.potentials = potentials
        self.center_predictors = center_predictors

        # Read and clean data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("'data' must be a pandas DataFrame.")

        # Some columns are converted to categorical
        self.data = with_categorical_cols(data, categorical)

        # Handle priors
        priors = {} if priors is None else deepcopy(priors)

        # Obtain design matrices and related objects.
        na_action = "drop" if dropna else "error"

        # Handle additional namespaces
        additional_namespace = transformations_namespace.copy()
        if not isinstance(extra_namespace, (type(None), dict)):
            raise ValueError("'namespace' must be a dictionary or None")

        if isinstance(extra_namespace, dict):
            additional_namespace.update(extra_namespace)

        # Create family
        self._set_family(family, link)

        ## Main component
        if isinstance(self.family, ORDINAL_FAMILIES):
            self.formula = check_ordinal_formula(self.formula)
            # Notice the intercept is added so formulae constrains categorical predictors, avoiding
            # linear dependencies with the cutpoints.
            # Then the intercept is removed from the design matrix because of the cutpoints.
            design = fm.design_matrices(
                self.formula.main + " + 1", self.data, na_action, 1, additional_namespace
            )
            design = remove_common_intercept(design)
        else:
            design = fm.design_matrices(
                self.formula.main, self.data, na_action, 1, additional_namespace
            )

        if design.response is None:
            raise ValueError(
                "No outcome variable is set! "
                "Please specify an outcome variable using the formula interface."
            )

        if self.family.likelihood.parent in priors:
            parent_priors = priors[self.family.likelihood.parent]
        else:
            parent_priors = priors

        # Add response component
        self.response_component = ResponseComponent(design.response, self)

        # Add component for parent parameter
        self.components[self.family.likelihood.parent] = DistributionalComponent(
            self.family.likelihood.parent, design, parent_priors, self, is_parent=True
        )

        # Get auxiliary parameters, so we add either distributional components or constant ones
        auxiliary_parameters = list(self.family.auxiliary_parameters)

        ## Other components
        ### Distributional
        for name, extra_formula in zip(self.formula.additionals_lhs, self.formula.additionals):
            # Check 'name' is part of parameter values
            if name not in auxiliary_parameters:
                raise ValueError(
                    f"'{name}' is not a parameter of the family."
                    f"Available parameters: {auxiliary_parameters}."
                )

            # Create design matrix, only for the response part
            design = fm.design_matrices(
                clean_formula_lhs(extra_formula), self.data, na_action, 1, additional_namespace
            )

            # If priors were not passed, pass an empty dictionary
            component_priors = priors.get(name, {})

            # Create distributional component
            self.components[name] = DistributionalComponent(
                name, design, component_priors, self, is_parent=False
            )

            # Remove parameter name from the list
            auxiliary_parameters.remove(name)

        ### Constant
        for name in auxiliary_parameters:
            component_prior = priors.get(name, None)
            self.components[name] = ConstantComponent(name, component_prior, self)

        # Build priors
        self._build_priors()

    def fit(
        self,
        draws=1000,
        tune=1000,
        discard_tuned_samples=True,
        omit_offsets=True,
        include_mean=None,
        include_response_params=False,
        inference_method="pymc",
        init="auto",
        n_init=50000,
        chains=None,
        cores=None,
        random_seed=None,
        **kwargs,
    ):
        """Fit the model using PyMC

        Parameters
        ----------
        draws : int, optional
            The number of samples to draw from the posterior distribution. Defaults to 1000.
        tune : int, optional
            Number of iterations to tune. Defaults to 1000. Samplers adjust the step sizes,
            scalings or similar during tuning. These tuning samples are be drawn in addition to the
            number specified in the `draws` argument, and will be discarded unless
            `discard_tuned_samples` is set to `False`.
        discard_tuned_samples : bool, optional
            Whether to discard posterior samples of the tune interval. Defaults to `True`.
        omit_offsets : bool, optional
            Omits offset terms in the `InferenceData` object returned when the model includes
            group specific effects. Defaults to `True`.
        include_mean : bool, optional, deprecated
            **This argument is deprecated and will be removed in future versions**.
            Use `include_response_params`.
        include_response_params : bool, optional
            Include parameters of the response distribution in the output. These usually take more
            space than other parameters as there's one of them per observation. Defaults to `False`.
        inference_method : str, optional
            The method to use for fitting the model. By default, `"pymc"`. This automatically
            assigns an MCMC method best suited for each kind of variables, like NUTS for continuous
            variables and Metropolis for non-binary discrete ones. NUTS implementations include
            `"pymc"`, `"nutpie"`, `"blackjax"`, and `"numpyro"`. Alternatively, `"vi"`, in which
            case the model will be fitted using variational inference as implemented in PyMC using
            the `fit` function. Finally, `"laplace"`, in which case a Laplace approximation is used
            and is not recommended other than for pedagogical use.
        init : str, optional
            Initialization method. Defaults to `"auto"`. The available methods are:

            - `"auto"`: Use `"jitter+adapt_diag"` and if this method fails it uses `"adapt_diag"`.
            - `"adapt_diag"`: Start with an identity mass matrix and then adapt a diagonal based on
            the variance of the tuning samples. All chains use the test value
            (usually the prior mean) as starting point.
            - `"jitter+adapt_diag"`: Same as `"adapt_diag"`, but use test value plus a uniform
            jitter in [-1, 1] as starting point in each chain.
            - `"advi+adapt_diag"`: Run ADVI and then adapt the resulting diagonal mass matrix based
            on the sample variance of the tuning samples.
            - `"advi+adapt_diag_grad"`: Run ADVI and then adapt the resulting diagonal mass matrix
            based on the variance of the gradients during tuning. This is **experimental** and might
            be removed in a future release.
            - `"advi"`: Run ADVI to estimate posterior mean and diagonal mass matrix.
            - `"advi_map"`: Initialize ADVI with MAP and use MAP as starting point.
            - `"map"`: Use the MAP as starting point. This is strongly discouraged.
            - `"adapt_full"`: Adapt a dense mass matrix using the sample covariances.
            All chains use the test value (usually the prior mean) as starting point.
            - `"jitter+adapt_full"`: Same as `"adapt_full"`, but use test value plus a uniform
            jitter in [-1, 1] as starting point in each chain.

        n_init : int, optional
            Number of initialization iterations. Only works for `"advi"` init methods.
        chains : int, optional
            The number of chains to sample. Running independent chains is important for some
            convergence statistics and can also reveal multiple modes in the posterior. If `None`,
            then set to either `cores` or 2, whichever is larger.
        cores : int, optional
            The number of chains to run in parallel. If `None`, it is equal to the number of CPUs
            in the system unless there are more than 4 CPUs, in which case it is set to 4.
        random_seed : int or list of ints, optional
            A list is accepted if cores is greater than one.
        kwargs : dict
            For other kwargs see the documentation for `PyMC.sample()`.

        Returns
        -------
        `InferenceData` or `Approximation`
            It returns an `InferenceData` if `inference_method` is `"pymc"`, `"nutpie"`,
            `"blackjax"`, `"numpyro"`, or `"laplace"`, and an `Approximation` object if  `"vi"`.
        """
        pass

    def build(self):
        """Set up the model for sampling/fitting

        Creates an instance of the underlying PyMC model and adds all the necessary terms to it.
        """
        self.backend = PyMCModel()
        self.backend.build(self)
        self.built = True

    def set_priors(self, priors=None, common=None, group_specific=None):
        """Set priors for one or more existing terms.

        Parameters
        ----------
        priors : dict or None, optional
            Dictionary of priors to update. Keys are names of terms to update; values are the new
            priors (either a `Prior` instance, or an int or float that scales the default priors).
        common : Prior, int, float or None, optional
            A prior specification to apply to all common terms included in the model.
        group_specific : Prior, int, float or None, optional
            A prior specification to apply to all group specific terms included in the model.
        """
        pass

    def _build_priors(self):
        """Carry out all operations related to the construction and/or scaling of priors."""
        pass

    def _set_priors(self, priors=None, common=None, group_specific=None):
        """Internal version of `set_priors()`, with same arguments.

        Runs during `Model._build_priors()`.
        """
        pass

    def _set_family(self, family, link):
        """Set the Family of the model

        Parameters
        ----------
        family : str or bambi.families.Family
            A specification of the model family.
            Either a string, or an instance of class `families.Family`.
            If a string is passed, a family with the corresponding name must be defined in the
            defaults loaded at model initialization.
        link : str or dict of str to str
            The name of the link function to use. Valid names are `"cloglog"`, `"identity"`,
            `"inverse_squared"`, `"inverse"`, `"log"`, `"logit"`, `"probit"`, and
            `"softmax"`. Not all the link functions can be used with all the families.
            If a dictionary, keys are the names of the target parameters and the values are the
            names of the link functions.

        Returns
        -------
        `None`
        """
        pass

    def set_alias(self, aliases):
        """Set aliases for the terms and auxiliary parameters in the model

        Parameters
        ----------
        aliases : dict of str to str
            A dictionary where key represents the original term name and the value is the alias.

        Returns
        -------
        `None`
        """
        pass

    def _check_built(self):
        # Checks if model is built, raises ValueError if not
        pass

    def plot_priors(
        self,
        draws=5000,
        var_names=None,
        filter_vars=None,
        kind="kde",
        ci_kind=None,
        ci_prob=None,
        point_estimate=None,
        plot_collection=None,
        backend=None,
        labeller=None,
        aes_by_visuals=None,
        visuals=None,
        stats=None,
        figsize=None,
        omit_offsets=True,
        omit_group_specific=True,
        random_seed=None,
        bins=None,
        hdi_prob=None,
        round_to=None,
        **pc_kwargs,
    ):
        """Samples from the prior distribution and plots its marginals.

        Parameters
        ----------
        draws : int, optional
            Number of draws to sample from the prior predictive distribution. Defaults to 5000.
        var_names : str or list of str, optional
            A list of names of variables for which to compute the prior predictive
            distribution. Defaults to `None` which means to include both observed and
            unobserved RVs.
        filter_vars : {"like", "regex"} or None, optional
            If `None`, interpret `var_names` as the real variable names.
            If `"like"`, interpret `var_names` as substrings of the real variable names.
            If `"regex"`, interpret `var_names` as regular expressions on the real variable names.
            Forwarded to [](`arviz_plots.plot_dist`).
        kind : str, optional
            Type of plot to display (`"kde"` or `"hist"`). For discrete variables this argument
            is ignored and a histogram is always used. Forwarded to [](`arviz_plots.plot_dist`).
        ci_kind : {"eti", "hdi"}, optional,
            Which credible interval to use. Defaults to `arviz_base.rcParams["stats.ci_kind"]`.
            Forwarded to [](`arviz_plots.plot_dist`).
        ci_prob : float, optional
            Indicates the probability that should be contained within the plotted credible interval.
            Defaults to `arviz_base.rcParams["stats.ci_prob"]`.
            Forwarded to [](`arviz_plots.plot_dist`).
        point_estimate : str, optional
            Plot point estimate per variable. Values should be `"mean"`, `"median"`, `"mode"`
            or `None`. When `None` (default) use `arviz_base.rcParams["stats.point_estimate"]`.
            Forwarded to [](`arviz_plots.plot_dist`).
        plot_collection : arviz_plots.PlotCollection, optional
            The plot collection to use. Forwarded to [](`arviz_plots.plot_dist`).
        backend : {"matplotlib", "plotly", "bokeh"}, optional
            The backend to use for plotting.
            If `None`, it inspects whether `plot_connection` is not `None`.
            If it's not, it uses `plot_collection.backend`.
            Otherweise, it uses `arviz_base.rcParams["plot.backend"]`.
            Forwarded to [](`arviz_plots.plot_dist`).
        labeller : arviz_base.labels.BaseLabeller, optional
            The labeller. If `None`, it uses [](`arviz_base.labels.BaseLabeller`).
            Forwarded to [](`arviz_plots.plot_dist`).
        aes_by_visuals : mapping of {str : sequence of str}, optional
            Forwarded to [](`arviz_plots.plot_dist`). See `aes_by_visuals` in there.
        visuals : mapping of {str : mapping or bool}, optional
            Forwarded to [](`arviz_plots.plot_dist`). See `visuals` in there.
        stats : mapping, optional
            Forwarded to [](`arviz_plots.plot_dist`). See `stats` in there.
        figsize : tuple, optional
            Figure size. If `None` it will be defined automatically.
        omit_offsets : bool
            Whether to omit offset terms in the plot. Defaults to `True`.
        omit_group_specific : bool, optional
            Whether to omit group specific effects in the plot. Defaults to `True`.
        random_seed : int or None, optional
            Seed for random number generator.
            Passed down to [Model.prior_predictive](`bambi.Model.prior_predictive`).
        bins : int, optional, deprecated
            **This argument is deprecated and will be removed in future versions**.
        hdi_prob : float or str, optional, deprecated
            Plots highest density interval for chosen percentage of density.
            Use `"hide"` to hide the highest density interval.
            **This argument is deprecated and will be removed in future versions**.
        round_to : int, optional, deprecated
            Controls formatting of floats. Defaults to 2 or the integer part, whichever is bigger.
            **This argument is deprecated and will be removed in future versions**.
        pc_kwargs : dict
            Passed to [](`arviz_plots.PlotCollection.wrap`)

        Returns
        -------
        pc : arviz_plots.PlotCollection

        """
        pass

    def prior_predictive(self, draws=500, var_names=None, omit_offsets=True, random_seed=None):
        """Generate samples from the prior predictive distribution.

        Parameters
        ----------
        draws : int, optional
            Number of draws to sample from the prior predictive distribution. Defaults to 500.
        var_names : str, list of str or None, optional
            A list of names of variables for which to compute the prior predictive distribution.
            Defaults to `None` which means both observed and unobserved RVs.
        omit_offsets : bool, optional
            Whether to omit offset terms in the plot. Defaults to `True`.
        random_seed : int or None, optional
            Seed for the random number generator.

        Returns
        -------
        InferenceData
            `InferenceData` object with the groups `prior`, `prior_predictive` and
            `observed_data`.
        """
        pass

    def predict(
        self,
        idata,
        kind="response_params",
        data=None,
        inplace=True,
        include_group_specific=True,
        sample_new_groups=False,
        random_seed=None,
    ):
        """Predict method for Bambi models

        Obtains in-sample and out-of-sample predictions from a fitted Bambi model.

        Parameters
        ----------
        idata : InferenceData
            The `InferenceData` instance returned by `.fit()`.
        kind : str, optional
            Indicates the type of prediction required. Can be `"response_params"` or
            `"response"`. The first returns draws from the posterior distribution of the
            likelihood parameters, while the latter returns the draws from the posterior
            predictive distribution (i.e. the posterior probability distribution for a new
            observation) in addition to the posterior distribution. Defaults to
            `"response_params"`.
        data : pd.DataFrame or None, optional
            An optional data frame with values for the predictors that are used to obtain
            out-of-sample predictions. If omitted, the original dataset is used.
        inplace : bool, optional
            If `True` it will modify `idata` in-place. Otherwise, it will return a copy of
            `idata` with the predictions added. If `kind="response_params"`, a new variable
            with the name of the parent parameter, e.g. `"mu"` and `"sigma"` for a Gaussian
            likelihood, or `"p"` for a Bernoulli likelihood, is added to the `posterior` group.
            If `kind="response"`, it appends a `posterior_predictive` group to `idata`. If
            any of these already exist, it will be overwritten.
        include_group_specific : bool, optional
            Determines if predictions incorporate group-specific effects. If `False`, predictions
            are made with common effects only (i.e. group specific are set to zero). Defaults to
            `True`.
        sample_new_groups : bool, optional
            Specifies if it is allowed to obtain predictions for new groups of group-specific terms.
            When `True`, each posterior sample for the new groups is drawn from the posterior
            draws of a randomly selected existing group. Since different groups may be selected at
            each draw, the end result represents the variation across existing groups.
            The method implemented is equivalent to `sample_new_levels="uncertainty"` in brms.
        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.

        Returns
        -------
        InferenceData or None
        """
        if kind not in ("mean", "pps", "response_params", "response"):
            raise ValueError("'kind' must be one of 'response_params' or 'response'")

        if kind == "mean":
            kind = "response_params"
            warnings.warn(
                "'mean' has been replaced by 'response_params' and "
                "is not going to work in the future",
                FutureWarning,
            )
        if kind == "pps":
            kind = "response"
            warnings.warn(
                "'pps' has been replaced by 'response' and is not going to work in the future",
                FutureWarning,
            )

        if not inplace:
            idata = deepcopy(idata)

        # Populate the posterior in the InferenceData object with the likelihood parameters
        idata = self._compute_likelihood_params(
            idata=idata,
            data=data,
            include_group_specific=include_group_specific,
            sample_new_groups=sample_new_groups,
            random_seed=random_seed,
        )

        # Only if requested predict the predictive distribution
        if kind == "response":
            response_aliased_name = get_aliased_name(self.response_component.term)
            required_kwargs = {
                "model": self,
                "posterior": idata.posterior,
                "random_seed": random_seed,
            }
            optional_kwargs = {"data": data}

            posterior_predictive = self.family.posterior_predictive(
                **required_kwargs, **optional_kwargs
            )
            posterior_predictive = posterior_predictive.to_dataset(name=response_aliased_name)

            if "posterior_predictive" in idata:
                del idata.posterior_predictive

            idata.add_groups({"posterior_predictive": posterior_predictive})
            idata.posterior_predictive = idata.posterior_predictive.assign_attrs(
                modeling_interface="bambi", modeling_interface_version=__version__
            )

        if inplace:
            return None
        else:
            return idata

    def r2_score(self, idata, summary=True):
        """R² for Bayesian regression models.

        The R², or coefficient of determination, is defined as the proportion of variance
        in the data that is explained by the model. It is computed as the variance of the
        predicted values divided by the variance of the predicted values plus the variance
        of the residuals. For details of the Bayesian R² see [1]_.

        Parameters
        ----------
        idata : InferenceData
            The `InferenceData` instance returned by `.fit()`. It should contain the
            `posterior_predictive` group, otherwise it will be computed and added to `idata`.
        summary : bool, optional
            If `True`, it returns a summary of the Bayesian R². Otherwise, it returns the
            posterior samples of the Bayesian R².

        Returns
        -------
        pandas.Series
            A series with the following indices:
            r2: mean value for the Bayesian R²
            r2_std: standard deviation of the Bayesian R².

        References
        ----------
        .. [1] Gelman et al. *R-squared for Bayesian regression models*.
            The American Statistician. 73(3) (2019). <https://doi.org/10.1080/00031305.2018.1549100>
        """
        pass

    def compute_log_likelihood(self, idata, data=None, inplace=True):
        """Compute the model's log-likelihood

        **NOTE**: This is a new feature and it may not work in all cases.

        Parameters
        ----------
        idata : InferenceData
            The `InferenceData` instance returned by `.fit()`.
        data : pd.DataFrame or None, optional
            An optional data frame with values for the predictors and the response on which
            the model's log-likelihood function is evaluated.
            If omitted, the original dataset is used.
        inplace : bool, optional
            If `True` it will modify `idata` in-place. Otherwise, it will return a copy of
            `idata` with the `log_likelihood` group added.

        Returns
        -------
        InferenceData or None
        """
        pass

    def _compute_likelihood_params(
        self,
        idata,
        data=None,
        include_group_specific=True,
        sample_new_groups=False,
        random_seed=None,
    ):
        """Computes the parameters of the likelihood (response distribution)

        This is a utility function that populates the posterior group in the InferenceData object
        with variables required by the model likelihood. This is useful for both posterior
        predictive sampling and the computation of the log-likelihood function.

        It returns the updated InferenceData object.
        """
        means_dict = {}
        hsgp_dict = {}  # To store the HSGP contributions (they are added to the posterior dataset)
        response_dim = "__obs__"

        for name, component in self.distributional_components.items():
            var_name = component.alias if component.alias else name
            means_dict[var_name] = component.predict(
                idata, data, include_group_specific, hsgp_dict, sample_new_groups, random_seed
            )

            # Drop var/dim if already present. Needed for out-of-sample predictions.
            if var_name in idata.posterior.data_vars:
                idata.posterior = idata.posterior.drop_vars(var_name)

        if response_dim in idata.posterior.dims:
            idata.posterior = idata.posterior.drop_dims(response_dim)

        # Use the first DataArray to get the number of observations
        obs_n = len(list(means_dict.values())[0].coords.get(response_dim))
        idata.posterior = idata.posterior.assign_coords({response_dim: list(range(obs_n))})

        for name, value in means_dict.items():
            idata.posterior[name] = value

        # Add HSGP contributions to the posterior dataset
        for component in self.distributional_components.values():
            for name, hsgp_contribution in hsgp_dict.items():
                term = component.hsgp_terms.get(name, None)
                if term is None:
                    continue
                term_aliased_name = get_aliased_name(term)
                idata.posterior[term_aliased_name] = hsgp_contribution.transpose(
                    "chain", "draw", ...
                )

        return idata

    def graph(self, formatting="plain", name=None, figsize=None, dpi=300, fmt="png"):
        """Produce a graphviz Digraph from a built Bambi model.

        Requires graphviz, which may be installed most easily with:
        ```cmd
        conda install -c conda-forge python-graphviz
        ```

        Alternatively, you may install the `graphviz` binaries yourself, and then
        `pip install graphviz` to get the python bindings.
        See <http://graphviz.readthedocs.io/en/stable/manual.html> for more information.

        Parameters
        ----------
        formatting : str, optional
            One of `"plain"` or `"plain_with_params"`. Defaults to `"plain"`.
        name : str, optional
            Name of the figure to save. Defaults to `None`, no figure is saved.
        figsize : tuple, optional
            Maximum width and height of figure in inches. Defaults to `None`, the figure size is
            set automatically. If defined and the drawing is larger than the given size, the drawing
            is uniformly scaled down so that it fits within the given size.  Only works if `name`
            is not `None`.
        dpi : int, optional
            Point per inch of the figure to save.
            Defaults to 300. Only works if `name` is not `None`.
        fmt : str, optional
            Format of the figure to save.
            Defaults to `"png"`. Only works if `name` is not `None`.

        Returns
        -------
        graphviz.Digraph
            The graph

        Examples
        --------
        ```python
        model = Model("y ~ x + (1|z)")
        model.fit()
        model.graph()
        ```
        """
        pass

    @property
    def formula(self):
        pass

    @formula.setter
    def formula(self, value):
        pass

    def __str__(self):
        # Empty list with the output components
        output_list = []

        # Build header
        parent_name = self.family.likelihood.parent
        formulas = self.formula.get_all_formulas()
        family_name = self.family.name
        parent_component = self.components[parent_name]

        links = [
            f"{key} = {value.name}"
            for key, value in self.family.link.items()
            if key == parent_name or key in self.distributional_components
        ]
        observations = self.response_component.term.data.shape[0]

        header_dict = {
            "Formula: ": formulas,
            "Family: ": family_name,
            "Link: ": links,
            "Observations: ": str(observations),
            "Priors: ": "",
        }

        width = 16
        spacer = "\n" + " " * width
        for key, value in header_dict.items():
            output_list.append(key.rjust(width) + spacer.join(listify(value)))

        # Build priors section. Make sure the parent component goes first.
        priors_dict = {parent_name: make_priors_summary(parent_component)}

        for name, component in self.distributional_components.items():
            if component.is_parent:
                continue
            priors_dict[name] = make_priors_summary(component)

        if self.constant_components:
            aux_str = "\n".join(
                [prior_repr(component) for component in self.constant_components.values()]
            )
            aux_str = "Auxiliary parameters\n" + wrapify(indentify(aux_str, 4), 100, 4)
            priors_dict[parent_name] = priors_dict[parent_name] + "\n\n" + aux_str

        for key, value in priors_dict.items():
            priors_dict[key] = indentify(value, 4)

        for key, value in priors_dict.items():
            output_list.append(indentify(f"target = {key}" + "\n" + value, 4))

        if self.backend and self.backend.fit:
            foot_list = [
                "------",
                "* To see a plot of the priors call the .plot_priors() method.",
                "* To see a summary or plot of the posterior pass the object returned by .fit() to "
                "az.summary() or az.plot_trace()",
            ]
            output_list.extend(foot_list)

        return "\n".join(output_list)

    def __repr__(self):
        return self.__str__()

    @property
    def constant_components(self):
        pass

    @property
    def distributional_components(self):
        pass


def with_categorical_cols(data: pd.DataFrame, columns) -> pd.DataFrame:
    """Convert selected columns of a DataFrame to categorical type.

    It converts all object columns plus columns specified in the `columns` argument.
    """
    pass


def prior_repr(term) -> str:
    """Get a string representation of a Bambi term."""
    pass


def hsgp_repr(term) -> str:
    """Get a string representation of a Bambi HSGP term."""
    pass


def make_priors_summary(component: DistributionalComponent) -> str:
    """Get a summary of terms and priors in a distributional component."""
    pass
