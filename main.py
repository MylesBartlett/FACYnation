from sys import argv

from utils import model_utils, config, data_loading, validation
import pylab as plt
import utils.plotting
import numpy as np
import models.models

_DEFAULT_CONFIG = 'run_configs/corr_bvg.ini'


if __name__ == '__main__':
    args = config.parse_arguments(argv[1] if len(argv) >= 2 else _DEFAULT_CONFIG)
    model = models.models.fetch_model(args.model)
    us_maize_regions = ['Indiana', 'Illinois', 'Ohio', 'Nebraska', 'Iowa', 'Minnesota']
    china_maize_regions_spring = ['Heilongjiang', 'Jilin', 'Liaoning']
    brazil_soybean = ['MatoGrosso']
    soybean_usa = ['Illinois', 'Indiana', 'Iowa', 'Minnesota', 'Missouri', 'NorthDakota', 'Ohio', 'SouthDakota']
    rice_mid_china = ['Anhui', 'Chongqing', 'Heilongjiang', 'Hubei', 'Hunan', 'Jiangsu', 'Jilin', 'Liaoning']

    data = data_loading.load_temp_precip_data('Maize', 'Spring', 'USA', us_maize_regions, month_start=0, month_end=12)
    save_path = f'models/saved_models/{args.model}_save'
    model_utils.save_model(model, save_path)
    # # # Load model to circumvent compile time
    load_path = f'{save_path}.pkl'
    model = model_utils.load_model(load_path)

    # cv_results = validation.time_series_cv(model, data, args)
    # mean_rrmse = np.mean(cv_results['test']['rrmse'])
    # mean_ns = np.mean(cv_results['test']['ns_eff'])
    # mean_explained_var = np.mean(cv_results['test']['explained_var'])
    # mean_r2 = np.mean(cv_results['test']['r2'])
    #
    # print('Mean RRMSE: ', mean_rrmse)
    # print('Mean NS:', mean_ns)
    # print('Mean Explained Var: ', mean_explained_var)
    # print('Mean R2: ', mean_r2)

    # cv_results = validation.leave_p_out_cv(model, data, args)
    # print(cv_results['prediction_errors'])

    fit = model.sampling(data, chains=args.chains, iter=args.iter,
                         verbose=args.verbose, seed=args.seed)
    # # print(fit)
    # ax = utils.plotting.plot_temp_precip_variation(fit, data)
    # ax.plot()
    # plt.show()
    # #
    # samples = fit.extract()
    utils.plotting.plot_per_region_yield_predictions(fit, us_maize_regions)
    plt.show()
    # #
    # samples = fit.extract()
    # pm = model_utils.extract_parameter_means(samples)
    # ax = utils.plotting.plot_temp_precip_constant_6m((0, 40), (0, 250), pm['mu_t'], pm['mu_p'], pm['sigma_t'],
    #                                pm['sigma_p'], [pm['norm']], pm['rho'])
    # ax.plot()
    # plt.show()
