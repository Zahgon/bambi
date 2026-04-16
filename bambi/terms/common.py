import numpy as np

import formulae.terms

from bambi.terms.base import BaseTerm


class CommonTerm(BaseTerm):
    """A common model term."""

    def __init__(self, term, prior, prefix=None):
        self.term = term
        self.prior = prior
        self.data = np.squeeze(term.data)
        self.prefix = prefix

        if self.categorical and len(self.levels) == 1 and (self.data == self.data[0]).all():
            raise ValueError(f"The term '{self.name}' has only 1 category!")

        if (
            not self.categorical
            and self.kind not in ("intercept", "offset")
            and np.all(self.data == self.data[0])
        ):
            raise ValueError(f"The term '{self.name}' is constant!")

    @property
    def term(self):
        pass

    @term.setter
    def term(self, value):
        pass

    @property
    def name(self):
        pass

    @property
    def coords(self):
        # Obtain pymc coordinates, only for categorical components of a term.
        # A categorical component can have up to two coordinates in the same model if it is
        # includied with both reduced and full rank encodings.
        pass

    @property
    def data(self):
        pass

    @data.setter
    def data(self, value):
        pass

    @property
    def kind(self):
        pass

    @property
    def shape(self):
        pass

    @property
    def categorical(self):
        # If the term has one component, it's categorical if the component is categorical.
        # If the term has more than one component (i.e. it is an interaction), it's categorical if
        # at least one of the components is categorical.
        pass

    @property
    def levels(self):
        pass

    def __str__(self):
        args = []
        if self.coords:
            args = [f"coords: {self.coords}"]
        return self.make_str(args)
