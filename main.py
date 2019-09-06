import os
from sys import argv
import _pickle
import os
from sys import argv

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import models.models
from utils import model_utils, config, data_loading, validation
from utils.model_utils import save_model, load_model

_DEFAULT_CONFIG = 'run_configs/corr_bvg.ini'


def print_metrics(cv_results):
    mean_rmse = np.mean(cv_results['test']['rmse'])
    mean_r2 = np.mean(cv_results['test']['r2'])

    print('Mean RMSE: ', mean_rmse)
    print('Mean R2: ', mean_r2)


def run(cv_method='loo'):
    args = config.parse_arguments(argv[1] if len(argv) >= 2 else _DEFAULT_CONFIG)
    # Load the data
    us_maize_regions = ['Indiana', 'Illinois', 'Ohio', 'Nebraska', 'Iowa',
                        'Minnesota']  # Growing season: April through to September

    data = data_loading.load_temp_precip_data('Maize', 'Spring', 'USA', us_maize_regions, range(3, 9))

    if args.model.lower() == 'corr_bvg' or args.model == 'uncorr_bvg':
        save_path = f'models/saved_models/{args.model}_save'
        load_path = f'{save_path}.pkl'
        if not os.path.exists(load_path):
            model = models.models.fetch_model(args.model)
            save_model(model=model, file_path=save_path)
        else:
            # Load model to circumvent compile time
            model = load_model(load_path)
        batched = False
        # Fit the model
        fit = model.sampling(data, chains=args.chains, iter=args.iter,
                             verbose=args.verbose, seed=args.seed)
        print(fit)
    elif args.model.lower() == 'gp':
        kernel = RBF()
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
        batched = True
    elif args.model.lower() == 'lr':
        model = LinearRegression()
        batched = True
    else:
        raise ValueError('Invalid model type.')

    if cv_method == 'rolling':
        # Rolling-origin cross-validation
        print("===> Rolling-origin CV")
        cv_results = validation.sliding_window_cv(model, data, args, batched=batched)
    elif cv_method == 'time-series':
        # Time-series cross validation, incrementing by one year each split
        print("===> Time-series CV")
        n_splits = 34
        cv_results = validation.time_series_cv(model, data, args, n_splits=n_splits, batched=batched)
    elif cv_method == 'loo':
        # LOO cross-validation
        print("===> LOO CV")
        cv_results = validation.leave_p_out_cv(model, data, args, p=1, batched=batched)
    else:
        # LTO cross-validation
        print("===> LTO CV")
        cv_results = validation.leave_p_out_cv(model, data, args, p=3, batched=batched)

    print_metrics(cv_results)


if __name__ == '__main__':
    run()
