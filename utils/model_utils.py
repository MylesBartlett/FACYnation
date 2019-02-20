import pickle
import numpy as np

from xidplus.stan_fit import stan_utility

_VALID_PARAMS = ['mu_t', 'mu_p', 'sigma_t', 'sigma_p', 'norm', 'rho']


def save_model(model, file_path):
    with open(f'{file_path}.pkl', 'wb') as f:
        pickle.dump(model, f)


def load_model(file_path):
    return pickle.load(open(file_path, 'rb'))


def extract_parameter_means(samples) -> dict:
    mean_parameters = {key: np.mean(val) for key, val in samples.items()
                       if key in _VALID_PARAMS}
    return mean_parameters


# This function should probably go somewhere else
def diagnose_fit(fit):
    stan_utility.check_div(fit)
    stan_utility.check_energy(fit)
    stan_utility.check_treedepth(fit)
