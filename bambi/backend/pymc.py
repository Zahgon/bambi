import functools
import logging
import traceback
import warnings
from copy import deepcopy
from importlib.metadata import version

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.util import get_default_varnames
from pytensor.tensor.special import softmax

from bambi.backend.links import (
    cloglog,
    identity,
    inverse_squared,
    logit,
    probit,
)
from bambi.backend.model_components import (
    ConstantComponent,
    DistributionalComponent,
    ResponseComponent,
)
from bambi.utils import get_aliased_name

_log = logging.getLogger("bambi")


__version__ = version("bambi")


_SUPPORTED_METHODS = {"pymc", "numpyro", "blackjax", "nutpie", "vi", "laplace"}
_DEPRECATION_MAP = {
    "mcmc": "pymc",
    "nuts_numpyro": "numpyro",
    "numpyro_nuts": "numpyro",
    "nuts_blackjax": "blackjax",
    "blackjax_nuts": "blackjax",
}


class PyMCModel:
    """PyMC model-fitting backend."""

    INVLINKS = {
        "cloglog": cloglog,
        "identity": identity,
        "inverse_squared": inverse_squared,
        "inverse": pt.reciprocal,
        "log": pt.exp,
        "logit": logit,
        "probit": probit,
        "softmax": functools.partial(softmax, axis=-1),
    }

    def __init__(self):
        self.name = pm.__name__
        self.version = pm.__version__
        self.vi_approx = None
        self.fit = False
        self.model = None
        self.spec = None
        self.components = {}
        self.response_component = None

    def build(self, spec):
        """Compile the PyMC model from an abstract model specification

        Parameters
        ----------
        spec : bambi.Model
            A Bambi `Model` instance containing the abstract specification of the model to compile.
        """
        self.model = pm.Model()
        self.components = {}

        for name, values in spec.response_component.term.coords.items():
            if name not in self.model.coords:
                self.model.add_coords({name: values})

        with self.model:
            # Add constant components
            for name, component in spec.constant_components.items():
                self.components[name] = ConstantComponent(component)
                self.components[name].build(self, spec)

            # Add distributional components
            for name, component in spec.distributional_components.items():
                self.components[name] = DistributionalComponent(component)
                self.components[name].build(self, spec)

            # Add response
            self.response_component = ResponseComponent(spec.response_component)
            self.response_component.build(self, spec)

            # Add potentials
            self.build_potentials(spec)

        self.spec = spec

    def run(
        self,
        draws=1000,
        tune=1000,
        discard_tuned_samples=True,
        omit_offsets=True,
        include_response_params=False,
        inference_method="pymc",
        init="auto",
        n_init=50000,
        chains=None,
        cores=None,
        random_seed=None,
        **kwargs,
    ):
        """Run PyMC sampler."""
        pass

    def build_potentials(self, spec):
        """Add potentials to the PyMC model

        Potentials are arbitrary quantities that are added to the model log likelihood.
        See 'Factor Potentials' in
        https://github.com/fonnesbeck/probabilistic_python/blob/main/pymc_intro.ipynb

        Parameters
        ----------
        spec : bambi.Model
            The model.
        """
        if spec.potentials is not None:
            count = 0
            for variable, constraint in spec.potentials:
                if isinstance(variable, (list, tuple)):
                    lambda_args = [self.model[var] for var in variable]
                    potential = constraint(*lambda_args)
                else:
                    potential = constraint(self.model[variable])
                pm.Potential(f"pot_{count}", potential)
                count += 1

    def _run_mcmc(
        self,
        draws,
        tune,
        discard_tuned_samples,
        omit_offsets,
        include_response_params,
        init,
        n_init,
        chains,
        cores,
        random_seed,
        sampler_backend,
        **kwargs,
    ):
        # Don't include the parameters of the likelihood, which are deterministics.
        # They can take lot of space in the trace and increase RAM requirements.
        pass

    def _clean_results(self, idata, omit_offsets, include_response_params):
        # Before doing anything, make sure we compute deterministics.
        # But, don't include those determinisics for parameters of the likelihood.

        pass

    def _run_vi(self, random_seed, **kwargs):
        pass

    def _run_laplace(self, draws, omit_offsets, include_response_params):
        """Fit a model using a Laplace approximation.

        Mainly for pedagogical use, provides reasonable results for approximately Gaussian
        posteriors. The approximation can be very poor for some models  like hierarchical ones.

        Parameters
        ----------
        draws : int
            The number of samples to draw from the posterior distribution.
        omit_offsets : bool
            Omits offset terms in the `InferenceData` object returned when the model includes
            group specific effects.
        include_response_params : bool
            Compute the posterior of the mean response.

        Returns
        -------
        An ArviZ's InferenceData object.
        """
        pass

    def _check_dependencies(self, inference_method):
        """Dependency checking given the selected inference method."""
        pass

    @property
    def constant_components(self):
        pass

    @property
    def distributional_components(self):
        pass


def _posterior_samples_to_idata(samples, model):
    """Create InferenceData from samples

    Parameters
    ----------
    samples : array
        Posterior samples
    model : PyMC model

    Returns
    -------
    An ArviZ's InferenceData object.
    """
    pass
