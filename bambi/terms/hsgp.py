# pylint: disable=no-member
from functools import partial

import numpy as np

import formulae.terms

from bambi.terms.base import BaseTerm, VALID_PRIORS

GP_VALID_PRIORS = tuple(value for value in VALID_PRIORS if value is not None)


# pylint: disable = invalid-name
class HSGPTerm(BaseTerm):
    def __init__(self, term, prior, prefix=None):
        """Create a term for an HSGP model component

        Parameters
        ----------
        term : formulae.terms.terms.Term
            A term that was created with `hsgp(...)`. The caller is an instance of `HSGP()`.
        prior : dict
            The keys are the names of the parameters of the covariance function and the values are
            instances of `bambi.Prior` or other values that are accepted by the covariance
            function.
        prefix : str
            It is used to indicate the term belongs to the component of a non-parent parameter.
            Defaults to `None`.
        """
        self.term = term
        self.prior = prior
        self.prefix = prefix
        self.hsgp_attributes = get_hsgp_attributes(term)
        self.hsgp = None
        properties_names = (
            "c",
            "by_levels",
            "cov",
            "share_cov",
            "scale",
            "iso",
            "centered",
            "drop_first",
            "variables_n",
            "groups_n",
            "mean",
            "maximum_distance",
        )
        self.__init_properties(properties_names)
        # When prior is none at initialization, then automatic priors are used
        self.automatic_priors = self.prior is None

    def __init_properties(self, names):
        """Initialize attributes as properties

        The properties are taken from the `self.hsgp_attributes` dictionary. This is to avoid
        writing many @property calls in the class definition.

        Parameters
        ----------
        names : sequence of str
            The names of the attributes taken from `self.hsgp_attributes`
        """
        pass

    @property
    def term(self):
        pass

    @term.setter
    def term(self, value):
        pass

    @property
    def data(self):
        pass

    @property
    def shape(self):
        pass

    @property
    def data_centered(self):
        pass

    @property
    def m(self):
        """Get the value of 'm', the number of basis vectors
        It's of shape (term.variables_n, ). It's computed by variable.
        """
        pass

    @property
    def L(self):
        """Get the value of L
        It's of shape (term.groups_n, term.variables_n). It's computed by variable and group.
        """
        pass

    @property
    def by(self):
        pass

    @property
    def prior(self):
        pass

    @prior.setter
    def prior(self, value):
        pass

    @property
    def scale_predictors(self):
        # If scale is None, look if it uses automatic priors.
        #  If automatic priors are used, it will scale the data
        #  If automatic priors are not used, it won't scale the data
        pass

    @property
    def coords(self):
        # This handles univariate and multivariate cases
        pass

    @property
    def name(self):
        pass

    @property
    def categorical(self):
        pass

    @property
    def levels(self):
        pass


def get_hsgp_attributes(term):
    """Extract HSGP attributes from a model matrix term

    Parameters
    ----------
    term : formulae.terms.terms.Term
        The formulae term that creates the HSGP term.

    Returns
    -------
    dict
        The attributes that will be passed to pm.gp.HSGP
    """
    pass
