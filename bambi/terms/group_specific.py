import numpy as np

import formulae.terms

from bambi.terms.base import BaseTerm, VALID_PRIORS
from bambi.priors.prior import Prior


class GroupSpecificTerm(BaseTerm):  # pylint: disable=too-many-instance-attributes
    def __init__(self, term, prior, prefix=None):
        self._hyperprior_alias = {}
        self.term = term
        self.prior = prior
        self.data = term.data
        self.group_index = self.invert_dummies(self.grouper)
        self.prefix = prefix

    def invert_dummies(self, dummies):
        """Invert dummies
        For the sake of computational efficiency (i.e., to avoid lots of large matrix
        multiplications in the backend), invert the dummy-coding process and represent full-rank
        dummies as a vector of indices into the coefficients.

        Only used when `bmb.config.SPARSE_DOT` is `False`.
        """
        pass

    @property
    def term(self):
        pass

    @term.setter
    def term(self, value):
        pass

    @property
    def coords(self):
        # The group is _always_ added as a coordinate. Maybe there's a cleaner way
        pass

    @property
    def data(self):
        pass

    @data.setter
    def data(self, value):
        pass

    @property
    def name(self):
        pass

    @property
    def kind(self):
        pass

    @property
    def shape(self):
        pass

    @property
    def categorical(self):
        # Determine if the expression is categorical
        pass

    @property
    def prior(self):
        pass

    @prior.setter
    def prior(self, value):
        # This does not check which argument has hyperprior (must be dispersion?)
        pass

    @property
    def groups(self):
        return self.term.groups

    @property
    def levels(self):
        pass

    @property
    def predictor(self):
        pass

    @property
    def grouper(self):
        pass

    @property
    def hyperprior_alias(self):
        pass

    @hyperprior_alias.setter
    def hyperprior_alias(self, values):
        pass

    def __str__(self):
        args = [f"groups: {self.groups}"]
        return self.make_str(args)
