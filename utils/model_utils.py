import pickle
import numpy as np
import random

# from xidplus.stan_fit import stan_utility

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


def sample_parameters(samples, num_samples_to_draw=20) -> dict:
    parameters = {}
    for key, val in samples.items():
        if key not in _VALID_PARAMS:
            continue
        parameters[key] = random.sample(list(val), num_samples_to_draw)

    return parameters
