"""
Aliased pySTAN models
"""
import pickle

import pystan
from xidplus.stan_fit import stan_utility


def uncorrelated_bivariate_gaussian_model() -> pystan.StanModel:
    """
    Parameterized bivariate normal distribution with
    no correlation between the variables.

    Returns:
        pySTAN model of an uncorrelated bivariate normal distribution.
    """
    return pystan.StanModel(file='./stan/2d-gaussian.stan')


def correlated_bivariate_gaussian_model() -> pystan.StanModel:
    """
    Parameterized bivariate normal distribution accounting for
     correlation between the variables.

    Returns:
        pySTAN model of an correlated bivariate normal distribution.
    """
    return pystan.StanModel(file='./stan/2d-gaussian_with_correlation.stan')


_MODELS = {
    'uncorr_bvg': uncorrelated_bivariate_gaussian_model,
    'corr_bvg': correlated_bivariate_gaussian_model
}


def fetch_model(model: str):
    return _MODELS[model]()

