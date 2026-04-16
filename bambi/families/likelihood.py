from collections import namedtuple

from bambi.utils import multilinify, indentify


DistSettings = namedtuple("DistSettings", ["params", "parent"])

DISTRIBUTIONS = {
    "Bernoulli": DistSettings(params=("p",), parent="p"),
    "Beta": DistSettings(params=("mu", "kappa"), parent="mu"),
    "BetaBinomial": DistSettings(params=("mu", "kappa"), parent="mu"),
    "Binomial": DistSettings(params=("p",), parent="p"),
    "Categorical": DistSettings(params=("p",), parent="p"),
    "Cumulative": DistSettings(params=("p", "threshold"), parent="p"),
    "DirichletMultinomial": DistSettings(params=("a",), parent="a"),
    "Gamma": DistSettings(params=("mu", "alpha"), parent="mu"),
    "Multinomial": DistSettings(params=("p",), parent="p"),
    "Normal": DistSettings(params=("mu", "sigma"), parent="mu"),
    "NegativeBinomial": DistSettings(params=("mu", "alpha"), parent="mu"),
    "Laplace": DistSettings(params=("mu", "b"), parent="mu"),
    "Poisson": DistSettings(params=("mu",), parent="mu"),
    "StudentT": DistSettings(params=("mu", "sigma", "nu"), parent="mu"),
    "VonMises": DistSettings(params=("mu", "kappa"), parent="mu"),
    "Wald": DistSettings(params=("mu", "lam"), parent="mu"),
    "ZeroInflatedBinomial": DistSettings(params=("p", "psi"), parent="p"),
    "ZeroInflatedNegativeBinomial": DistSettings(params=("mu", "alpha", "psi"), parent="mu"),
    "ZeroInflatedPoisson": DistSettings(params=("mu", "psi"), parent="mu"),
}


class Likelihood:
    """Representation of a Likelihood function for a Bambi model

    Parameters
    ----------
    name : str
        Name of the likelihood function. Must be a valid PyMC distribution name.
    params : sequence of str or None, optional
        The name of the parameters the likelihood function accepts.
    parent : str or None, optional
        Optional specification of the name of the mean parameter in the likelihood.
        This is the parameter whose transformation is modeled by the linear predictor.
    dist : pymc.Distribution or callable or None, optional
        Optional custom PyMC distribution that will be used to compute the likelihood.

    Notes
    -----
    - `parent` must be in `params`
    - `parent` is inferred from the `name` if it is a known name
    """

    DISTRIBUTIONS = DISTRIBUTIONS

    def __init__(self, name, params=None, parent=None, dist=None):
        self.name = name
        self.params = params
        self.parent = parent
        self.dist = dist

    @property
    def params(self):
        pass

    @params.setter
    def params(self, value):
        pass

    @property
    def parent(self):
        pass

    @parent.setter
    def parent(self, value):
        # Checks are made when using a known distribution
        pass

    def __str__(self):
        args = [
            ("name", self.name),
            ("params", self.params),
            ("parent", self.parent),
            ("dist", self.dist),
        ]
        args = [f"{arg[0]}: {arg[1]}" for arg in args]
        return f"{self.__class__.__name__}({indentify(multilinify(args))}\n)"

    def __repr__(self):
        return self.__str__()
