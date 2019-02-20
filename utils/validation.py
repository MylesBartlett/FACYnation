from sklearn import model_selection as ms
from sklearn import metrics

from utils import model_utils, evaluate
from utils.data_loading import extract_data_by_year_index


def _cross_validate(model, data, cross_validator, args):

    cv_results = {'predicted_yields': [],
                  'actual_yields': [],
                  'prediction_errors': []}

    for i, (train_index, test_index) in enumerate(cross_validator.split(data['d_yields'][0])):
        train_data = extract_data_by_year_index(data, train_index)
        test_data = extract_data_by_year_index(data, test_index)

        fit = model.sampling(train_data, chains=args.chains, iter=args.iter,
                             verbose=args.verbose, seed=args.seed)
        samples = fit.extract()
        param_means = model_utils.extract_parameter_means(samples)
        y_pred = evaluate.compute_annual_yield_anom_6m_tp(test_data, param_means['mu_t'], param_means['mu_p'],
                                                          param_means['sigma_t'], param_means['sigma_p'],
                                                          [param_means['norm']],
                                                          param_means['rho'] if 'rho' in param_means else None,
                                                          average=False)
        actual_y = test_data['d_yields']
        cv_results['predicted_yields'].append(y_pred)
        cv_results['actual_yields'].append(actual_y)

        error = metrics.mean_squared_error(y_pred, actual_y)

        cv_results['prediction_errors'].append(error)
        print(f'====> Fold {i+1} validation Error: {error:.4f}')

    return cv_results


def time_series_cv(model, data, args, n_splits=5):
    cross_validator = ms.TimeSeriesSplit(n_splits=n_splits)
    return _cross_validate(model, data, cross_validator, args)


def leave_p_out_cv(model, data, args, p=3):
    cross_validator = ms.LeavePOut(p=p)
    return _cross_validate(model, data, cross_validator, args)
