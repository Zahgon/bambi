import formulae.terms

from bambi.terms.base import BaseTerm
from bambi.terms.utils import is_response_of_kind


class ResponseTerm(BaseTerm):
    def __init__(self, response, family):
        self.term = response.term.term
        self.family = family
        self.is_censored = is_response_of_kind(self.term, "censored")
        self.is_constrained = is_response_of_kind(self.term, "constrained")
        self.is_truncated = is_response_of_kind(self.term, "truncated")
        self.is_weighted = is_response_of_kind(self.term, "weighted")

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
    def name(self):
        pass

    @property
    def shape(self):
        pass

    @property
    def levels(self):
        pass

    @property
    def categorical(self):
        pass

    @property
    def reference(self):
        pass

    @property
    def coords(self):
        pass

    @property
    def success(self):
        pass

    @property
    def binary(self):
        # Maybe it's not needed here anymore
        pass

    def __str__(self):
        extras = []
        if self.categorical:
            if self.binary:
                extras += [f"success: {self.success}"]
            else:
                extras += [f"reference: {self.reference}"]
        return self.make_str(extras)
