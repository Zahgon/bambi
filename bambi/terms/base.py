from abc import ABC, abstractmethod

from bambi.priors.prior import Prior
from bambi.utils import indentify, multilinify


VALID_PRIORS = (Prior, int, float, type(None))


class BaseTerm(ABC):
    _alias = None
    _prior = None

    @property
    @abstractmethod
    def term(self): ...

    @property
    @abstractmethod
    def data(self): ...

    @property
    @abstractmethod
    def name(self): ...

    @property
    @abstractmethod
    def shape(self): ...

    @property
    @abstractmethod
    def levels(self): ...

    @property
    @abstractmethod
    def categorical(self): ...

    @property
    def alias(self):
        pass

    @alias.setter
    def alias(self, value):
        pass

    @property
    def prior(self):
        pass

    @prior.setter
    def prior(self, value):
        pass

    @property
    def ndim(self):
        pass

    def make_str(self, extras=None):
        pass

    def __str__(self):
        return self.make_str()

    def __repr__(self):
        return self.__str__()
