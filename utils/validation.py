from sklearn import model_selection as ms
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

from utils import model_utils, evaluate
from utils.data_loading import extract_data_by_year_index
import utils.metrics
import numpy as np


def _cross_validate(model, data, cross_validator, args):

    cv_values = ['predicted_yields', 'actual_yields', 'rmse', 'rrmse', 'ns_eff', 'explained_var', 'r2']
    cv_results = {'train': {key: [] for key in cv_values},
                  'test': {key: [] for key in cv_values}}

    for i, (train_index, test_index) in enumerate(cross_validator.split(data['d_yields'][0])):
        train_data = extract_data_by_year_index(data, train_index)
        test_data = extract_data_by_year_index(data, test_index)

        fit = model.sampling(train_data, chains=args.chains, iter=args.iter,
                             verbose=args.verbose, seed=args.seed)
        samples = fit.extract()
        param_means = model_utils.extract_parameter_means(samples)

        rmse_train, rrmse_train, ns_eff_train, explained_var_train, r2_train = _cv_evaluate(param_means, train_data,
                                                                                            cv_results['train'])

        rmse_test, rrmse_test, ns_eff_test, explained_var_test, r2_test = _cv_evaluate(param_means, test_data,
                                                                                       cv_results['test'])

        print(f'====> Fold {i+1} validation\nRMSE: {rmse_test}\nRRMSE: {rrmse_test:.4f}\nNS: {ns_eff_test:.4f}\n'
              f'Explained var: {explained_var_test:.4f}\n{rrmse_test:.4f}')

    return cv_results


def _cv_evaluate(param_means, data, cv_dict):
    y_pred = evaluate.compute_annual_yield_anom_tp(data, param_means['mu_t'], param_means['mu_p'],
                                                   param_means['sigma_t'], param_means['sigma_p'],
                                                   [param_means['norm']],
                                                   param_means['rho'] if 'rho' in param_means else None,
                                                   average=False)
    y_true = data['d_yields']
    rmse = utils.metrics.rmse(y_pred, y_true)
    rrmse = utils.metrics.rrmse(y_pred, y_true)
    ns_eff = utils.metrics.nash_sutcliffe_eff(y_pred, y_true)
    explained_var = explained_variance_score(y_pred, y_true)
    r2 = r2_score(y_pred, y_true)

    cv_dict['predicted_yields'].extend(np.nanmean(y_pred, axis=0).flatten())
    cv_dict['actual_yields'].extend(np.nanmean(y_true, axis=0).flatten())

    cv_dict['rmse'].append(rmse)
    cv_dict['rrmse'].append(rrmse)
    cv_dict['ns_eff'].append(ns_eff)
    cv_dict['explained_var'].append(explained_var)
    cv_dict['r2'].append(r2)

    return rmse, rrmse, ns_eff, explained_var, r2


class CenteredWindowSplit:

    def __init__(self, radius=1):
        self.radius = radius

    def split(self, data):
        n_samples = len(data)
        for test_index in range(n_samples):
            window = range(max(0, test_index-self.radius), min(test_index+self.radius+1, n_samples))
            train_indexes = list(set(range(n_samples)) - set(window))
            yield train_indexes, [test_index]


def time_series_cv(model, data, args, n_splits=5):
    cross_validator = ms.TimeSeriesSplit(n_splits=n_splits)
    return _cross_validate(model, data, cross_validator, args)


def leave_p_out_cv(model, data, args, p=3):
    cross_validator = ms.LeavePOut(p=p)
    return _cross_validate(model, data, cross_validator, args)


def sliding_window_cv(model, data, args, r=1):
    cross_validator = CenteredWindowSplit(radius=r)
    return _cross_validate(model, data, cross_validator, args)
