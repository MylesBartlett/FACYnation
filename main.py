import logging
import os
import argparse
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

import models.models
from utils import config, data_loading, validation
from utils.model_utils import save_model, load_model

logging.getLogger("pystan").propagate = False


def arguments():
    parser = argparse.ArgumentParser()
    # General data set settings
    parser.add_argument("--anom-type", type=str,
                         choices=["mean", "cmnist", "celeba"], default="mean")
    parser.add_argument("cv-method", type=str,
                         choices=["loo", "lto", "rolling", "time-series"], default="loo")
    parser.add_argument("model", type=str,
                        choices=["corr_bvg", "lr", "gp"], default="corr_bvg")
    
    return parser.parse_args()
    

def print_metrics(cv_results):
    mean_rmse = np.mean(cv_results['test']['rmse'])
    mean_r2 = np.mean(cv_results['test']['r2'])

    print('Mean RMSE: ', mean_rmse)
    print('Mean R2: ', mean_r2)


def run(cv_method='loo', anom_type='mean', model="corr_bvg"):
    args = f"run_configs/{model}.ini"
    # Load the data
    us_maize_regions = ['Indiana', 'Illinois', 'Ohio', 'Nebraska', 'Iowa',
                        'Minnesota']  # Growing season: April through to September

    data = data_loading.load_temp_precip_data('Maize', 'Spring', 'USA', us_maize_regions,
                                              range(3, 9), anom_type=anom_type)

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
    elif args.model.lower() == 'gp':
        kernel = RBF(length_scale=0.5)
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
        batched = True
    elif args.model.lower() == 'lr':
        # model = LinearRegression(fit_intercept=True, normalize=True)
        model = RidgeCV()
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

    # import pandas as pd
    #
    # test_preds = cv_results['test']['predicted_yields']
    # test_preds_df = pd.DataFrame(test_preds)
    # test_preds_df.to_csv(f"{args.model}_{cv_method}_{anom_type}_preds")
    #
    # test_observed = cv_results['test']['actual_yields']
    # test_observed_df = pd.DataFrame(test_observed)
    # test_observed_df.to_csv(f"{args.model}_{cv_method}_{anom_type}_observed")
    #
    # plt.scatter(np.reshape(cv_results['test']['actual_yields'], -1),
    #             np.reshape(cv_results['test']['predicted_yields'], -1),
    #             label='Predicted', s=20,
    #             edgecolors=(0, 0, 0))
    # plt.plot(cv_results['test']['actual_yields'], cv_results['test']['actual_yields'],
    #          color='k', label='Actual', alpha=0.3)
    # # plt.plot(np.arange(1980, 2015), cv_results['test']['actual_yields'],
    # #          marker='s', color='k', label='Observed Yield')
    # plt.xlabel('Actual yield (tonnes ha$^{-1}$)')
    # plt.ylabel('Predicted yield (tonnes ha$^{-1}$)')
    # # plt.legend()
    # plt.savefig(f'loo_{args.model}_{cv_method}_{anom_type}.pdf')
    # plt.savefig(f'{args.model}_{cv_method}_{anom_type}.png')
    # plt.show()


if __name__ == '__main__':
    args = arguments()
    run(cv_method=args.cv_method, anom_type=args.anom_type, model=args.model)
