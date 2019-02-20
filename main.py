from sys import argv

from utils import model_utils, config, data_loading, validation
import pylab as plt
import utils.plotting

_DEFAULT_CONFIG = 'run_configs/corr_bvg.ini'


if __name__ == '__main__':
    args = config.parse_arguments(argv[1] if len(argv) >= 2 else _DEFAULT_CONFIG)
    # model = models.fetch_model(args.model)

    data = data_loading.prepare_us_maize_data()
    save_path = f'models/saved_models/{args.model}_save'
    # model_utils.save_model(model, save_path)
    # Load model to circumvent compile time
    load_path = f'{save_path}.pkl'
    model = model_utils.load_model(load_path)

    cv_results = validation.time_series_cv(model, data, args)
    print(cv_results['test']['rrmse'])

    # cv_results = validation.leave_p_out_cv(model, data, args)
    # print(cv_results['prediction_errors'])

    # fit = model.sampling(data, chains=args.chains, iter=args.iter,
    #                      verbose=args.verbose, seed=args.seed)
    # print(fit)
    # #
    # samples = fit.extract()
    # utils.plotting.plot_per_region_yield_predictions(fit, data_loading.US_MAIZE_STATES)
    # plt.show()
    #
    # pm = model_utils.extract_parameter_means(samples)
    # ax = plotting.plot_temp_precip_grid((0, 40), (0, 250), pm['mu_t'], pm['mu_p'], pm['sigma_t'],
    #                                pm['sigma_p'], [pm['norm']], pm['rho'])
    # ax.plot()
    # plt.show()
