# pylint: disable=unused-argument
import numpy as np
import pytensor.tensor as pt
import xarray as xr

from bambi.families.family import Family
from bambi.transformations import transformations_namespace
from bambi.utils import extract_argument_names, get_aliased_name, response_evaluate_new_data


class MultivariateFamily(Family):
    KIND = "Multivariate"


class Multinomial(MultivariateFamily):
    SUPPORTED_LINKS = {"p": ["softmax"]}
    INVLINK_KWARGS = {"axis": -1}

    @staticmethod
    def transform_linear_predictor(
        model, linear_predictor: xr.DataArray, posterior: xr.DataArray
    ) -> xr.DataArray:  # pylint: disable = unused-variable
        response_name = get_aliased_name(model.response_component.term)
        response_levels_dim = response_name + "_reduced_dim"
        linear_predictor = linear_predictor.pad({response_levels_dim: (1, 0)}, constant_values=0)
        return linear_predictor

    def transform_coords(self, model, mean):
        # The mean has the reference level in the dimension, a new name is needed
        response_name = get_aliased_name(model.response_component.term)
        response_levels_dim = response_name + "_reduced_dim"
        response_levels_dim_complete = response_name + "_dim"
        levels_complete = model.response_component.term.levels
        mean = mean.rename({response_levels_dim: response_levels_dim_complete})
        mean = mean.assign_coords({response_levels_dim_complete: levels_complete})
        return mean

    def posterior_predictive(self, model, posterior, random_seed, **kwargs):
        data = kwargs["data"]
        if data is None:
            y = model.response_component.term.data
            trials = model.response_component.term.data.sum(1).astype(int)
        else:
            y = response_evaluate_new_data(model, data).astype(int)
            trials = y.sum(1).astype(int)

        # Prepend 'draw' and 'chain' dimensions
        trials = trials[np.newaxis, np.newaxis, :]
        dont_reshape = ["n"]
        return super().posterior_predictive(
            model, posterior, n=trials, dont_reshape=dont_reshape, random_seed=random_seed
        )

    def log_likelihood(self, model, posterior, data, **kwargs):
        pass

    def get_coords(self, response):
        # For the moment, it always uses the first column as reference.
        pass

    def get_levels(self, response):
        pass

    @staticmethod
    def transform_backend_kwargs(kwargs):
        kwargs["n"] = kwargs["observed"].sum(axis=1).astype(int)
        return kwargs

    @staticmethod
    def transform_backend_eta(eta, kwargs):
        data = kwargs["observed"]

        # Add column of zeros to the linear predictor for the reference level (the first one)
        shape = (data.shape[0], 1)

        # The first line makes sure the intercept-only models work
        eta = np.ones(shape) * eta  # (response_levels, ) -> (n, response_levels)
        eta = pt.concatenate([np.zeros(shape), eta], axis=1)
        return eta


class DirichletMultinomial(MultivariateFamily):
    SUPPORTED_LINKS = {"a": ["log"]}

    def posterior_predictive(self, model, posterior, random_seed, **kwargs):
        data = kwargs["data"]
        if data is None:
            y = model.response_component.term.data
            trials = model.response_component.term.data.sum(1).astype(int)
        else:
            y = response_evaluate_new_data(model, data).astype(int)
            trials = y.sum(1).astype(int)

        # Prepend 'draw' and 'chain' dimensions
        trials = trials[np.newaxis, np.newaxis, :]
        dont_reshape = ["n"]
        return super().posterior_predictive(
            model, posterior, n=trials, dont_reshape=dont_reshape, random_seed=random_seed
        )

    def log_likelihood(self, model, posterior, data, **kwargs):
        pass

    def get_coords(self, response):
        pass

    def get_levels(self, response):
        pass

    @staticmethod
    def transform_backend_kwargs(kwargs):
        kwargs["n"] = kwargs["observed"].sum(axis=1).astype(int)
        return kwargs
