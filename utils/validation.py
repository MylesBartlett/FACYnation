from sklearn import model_selection as ms
from sklearn.metrics import mean_squared_error

from utils import model_utils, evaluate
from utils.data_loading import extract_data_by_year_index
import utils.metrics


def _cross_validate(model, data, cross_validator, args):

    cv_values = ['predicted_yields', 'actual_yields', 'rrmse', 'ns_eff']
    cv_results = {'train': dict.fromkeys(cv_values, []),
                  'test': dict.fromkeys(cv_values, [])}

    # metrics = [utils.metrics.rrmse, utils.metrics.nash_sutcliffe_eff]

    for i, (train_index, test_index) in enumerate(cross_validator.split(data['d_yields'][0])):
        train_data = extract_data_by_year_index(data, train_index)
        test_data = extract_data_by_year_index(data, test_index)

        fit = model.sampling(train_data, chains=args.chains, iter=args.iter,
                             verbose=args.verbose, seed=args.seed)
        samples = fit.extract()
        param_means = model_utils.extract_parameter_means(samples)

        rmse_train, ns_eff_train = _cv_evaluate(param_means, train_data, cv_results['train'])
        rmse_test, ns_eff_test = _cv_evaluate(param_means, test_data, cv_results['test'])

        print(f'====> Fold {i+1} validation RMSE: {rmse_test:.4f}')

    return cv_results


def _cv_evaluate(param_means, data, cv_dict):
    y_pred = evaluate.compute_annual_yield_anom_6m_tp(data, param_means['mu_t'], param_means['mu_p'],
                                                      param_means['sigma_t'], param_means['sigma_p'],
                                                      [param_means['norm']],
                                                      param_means['rho'] if 'rho' in param_means else None,
                                                      average=False)
    y_true = data['d_yields']
    rmse = utils.metrics.rrmse(y_pred, y_true)
    ns_eff = utils.metrics.nash_sutcliffe_eff(y_pred, y_true)
    cv_dict['rrmse'].append(rmse)
    cv_dict['ns_eff'].append(ns_eff)
    return rmse, ns_eff


def time_series_cv(model, data, args, n_splits=5):
    cross_validator = ms.TimeSeriesSplit(n_splits=n_splits)
    return _cross_validate(model, data, cross_validator, args)


def leave_p_out_cv(model, data, args, p=3):
    cross_validator = ms.LeavePOut(p=p)
    return _cross_validate(model, data, cross_validator, args)
