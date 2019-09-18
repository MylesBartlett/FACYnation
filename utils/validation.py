from sklearn import model_selection as ms
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

from utils import model_utils, evaluate, metrics
from utils.data_loading import extract_data_by_year_index, batch_data
import numpy as np
import random


def _cross_validate_batched(model, data, cross_validator):

    cv_results = {'test':
        {
            'rmse': [],
            'r2': [],
            'predicted_yields': [],
            'actual_yields': []}
    }

    for i, (train_index, test_index) in enumerate(cross_validator.split(data['d_yields'][0])):
        train_data = extract_data_by_year_index(data, train_index)
        test_data = extract_data_by_year_index(data, test_index)
        X_train, y_train = batch_data(train_data).values()
        X_test, y_test = batch_data(test_data).values()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = metrics.gelman_r2(y_true=y_test, y_pred=y_pred)

        cv_results['test']['rmse'].append(mean_squared_error(y_pred, y_test)**0.5)
        cv_results['test']['r2'].append(r2)
        cv_results['test']['predicted_yields'].append(y_pred)
        cv_results['test']['actual_yields'].append(y_test)

    return cv_results


def _cross_validate(model, data, cross_validator, args, use_mean_params=False):

    cv_values = ['predicted_yields', 'actual_yields', 'rmse', 'rrmse', 'ns_eff', 'explained_var', 'r2']
    cv_results = {'train': {key: [] for key in cv_values},
                  'test': {key: [] for key in cv_values}}

    for i, (train_index, test_index) in enumerate(cross_validator.split(data['d_yields'][0])):
        train_data = extract_data_by_year_index(data, train_index)
        test_data = extract_data_by_year_index(data, test_index)

        fit = model.sampling(train_data, chains=args.chains, iter=args.iter,
                             verbose=args.verbose, seed=args.seed)
        samples = fit.extract()
        if use_mean_params:
            params = model_utils.extract_parameter_means(samples)
        else:
            params = model_utils.sample_parameters(samples, num_samples_to_draw=1)

        _cv_evaluate(params, test_data, cv_results['test'], use_mean_params=use_mean_params)

        print(cv_results['test']['rmse'])

        print(
            f"====> Fold {i+1} "
            f"validation\nRMSE: {cv_results['test']['rmse'][-1]:.4f}\n"
            f"R2: {cv_results['test']['r2'][-1]:.4f}\n")

    return cv_results


def _cv_evaluate(param_means, data, cv_dict, use_mean_params=True):
    if use_mean_params:
        y_pred = evaluate.compute_annual_yield_anom_tp(data, param_means['mu_t'], param_means['mu_p'],
                                                       param_means['sigma_t'], param_means['sigma_p'],
                                                       [param_means['norm']],
                                                       param_means['rho'] if 'rho' in param_means else None,
                                                       average=False)

        y_true = data['d_yields']
        rmse = metrics.rmse(y_true=y_true, y_pred=y_pred)

        r2 = metrics.gelman_r2(y_true=y_true, y_pred=y_pred)

        cv_dict['predicted_yields'].extend(y_pred)
        cv_dict['actual_yields'].extend(y_true)

        cv_dict['rmse'].append(rmse)
        cv_dict['r2'].append(r2)

    else:
        r2s, rmses = [], []
        for i in range(len(param_means['mu_t'])):
            y_pred = evaluate.compute_annual_yield_anom_tp(data, param_means['mu_t'][i], param_means['mu_p'][i],
                                                           param_means['sigma_t'][i], param_means['sigma_p'][i],
                                                           [param_means['norm']][i],
                                                           param_means['rho'][i] if 'rho' in param_means else None,
                                                           average=False)
            y_true = data['d_yields']

            cv_dict['predicted_yields'].extend(y_pred)
            cv_dict['actual_yields'].extend(y_true)

            rmse = metrics.rmse(y_true=y_true, y_pred=y_pred)
            r2 = metrics.gelman_r2(y_true=y_true, y_pred=y_pred)

            r2s.append(r2)
            rmses.append(rmse)

        cv_dict['rmse'].append(np.mean(rmses))
        cv_dict['r2'].append(np.median(r2s))


class CenteredWindowSplit:

    def __init__(self, radius=1):
        self.radius = radius

    def split(self, data):
        n_samples = len(data)
        for test_index in range(n_samples):
            window = range(max(0, test_index-self.radius), min(test_index+self.radius+1, n_samples))
            train_indexes = list(set(range(n_samples)) - set(window))
            yield train_indexes, [test_index]


class RandomSplit:

    def __init__(self, n_splits=10, test_size=0.2, seed=42):
        assert(0 < test_size < 1)
        self.n_splits = n_splits
        self.test_size = test_size
        self.seed = seed

    def split(self, data):
        n_samples = len(data)
        n_test = int(self.test_size * n_samples)
        random.seed(self.seed)
        for i in range(self.n_splits):
            test_indexes = random.sample(range(n_samples), n_test)
            train_indexes = list(set(range(n_samples)) - set(test_indexes))
            yield train_indexes, test_indexes


def time_series_cv(model, data, args, n_splits=5, batched=False):
    cross_validator = ms.TimeSeriesSplit(n_splits=n_splits)
    if batched:
        return _cross_validate_batched(model, data, cross_validator)
    else:
        return _cross_validate(model, data, cross_validator, args)


def leave_p_out_cv(model, data, args, p=3, batched=False):
    cross_validator = ms.LeavePOut(p=p)
    if batched:
        return _cross_validate_batched(model, data, cross_validator)
    else:
        return _cross_validate(model, data, cross_validator, args)


def sliding_window_cv(model, data, args, r=1, batched=False):
    cross_validator = CenteredWindowSplit(radius=r)
    if batched:
        return _cross_validate_batched(model, data, cross_validator)
    else:
        return _cross_validate(model, data, cross_validator, args)


def random_cv(model, data, args, n_splits=10, test_size=0.2, seed=42, batched=False):
    cross_validator = RandomSplit(n_splits=n_splits, test_size=test_size, seed=seed)
    if batched:
        return _cross_validate_batched(model, data, cross_validator)
    else:
        return _cross_validate(model, data, cross_validator, args)
