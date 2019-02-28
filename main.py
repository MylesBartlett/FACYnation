from sys import argv

from utils import model_utils, config, data_loading, validation
import pylab as plt
import utils.plotting
import numpy as np
import models.models

_DEFAULT_CONFIG = 'run_configs/corr_bvg.ini'


if __name__ == '__main__':
    args = config.parse_arguments(argv[1] if len(argv) >= 2 else _DEFAULT_CONFIG)
    # model = models.models.fetch_model(args.model)
    us_maize_regions = ['Indiana', 'Illinois', 'Ohio', 'Nebraska', 'Iowa', 'Minnesota']  # Growing season: April through to September
    china_spring_maize_regions = ['Heilongjiang', 'Jilin', 'Liaoning']  # need to address nans
    brazil_soybean = ['MatoGrosso']
    soybean_usa = ['Illinois', 'Indiana', 'Iowa', 'Minnesota', 'Missouri', 'NorthDakota', 'Ohio', 'SouthDakota']
    wheat_winter_usa = ['Colorado', 'Idaho', 'Kansas', 'Montana', 'Nebraska', 'Ohio', 'Oklahoma', 'SouthDakota', 'Texas', 'Washington']

    rice_mid_china = ['Anhui', 'Chongqing', 'Heilongjiang', 'Hubei', 'Hunan', 'Jiangsu', 'Jilin', 'Liaoning']

    data = data_loading.load_temp_precip_data('Maize', 'Spring', 'USA', us_maize_regions, range(3, 9))
    save_path = f'models/saved_models/{args.model}_save'

    # # Load model to circumvent compile time
    load_path = f'{save_path}.pkl'
    model = model_utils.load_model(load_path)

    fit = model.sampling(data, chains=args.chains, iter=args.iter,
                         verbose=args.verbose, seed=args.seed)
    print(fit)

    cv_results = validation.sliding_window_cv(None, data, args)
    # # n_splits = 34
    # cv_results = validation.leave_p_out_cv(model, data, args, p=1)
    #
    mean_rmse = np.mean(cv_results['test']['rmse'])
    mean_rrmse = np.mean(cv_results['test']['rrmse'])
    mean_ns = np.mean(cv_results['test']['ns_eff'])
    mean_explained_var = np.mean(cv_results['test']['explained_var'])
    mean_r2 = np.mean(cv_results['test']['r2'])

    print('Mean RMSE: ', mean_rmse)
    print('Mean RRMSE: ', mean_rrmse)
    print('Mean NS:', mean_ns)
    print('Mean Explained Var: ', mean_explained_var)
    print('Mean R2: ', mean_r2)

    print(cv_results['test']['predicted_yields'])
    print(cv_results['test']['actual_yields'])
    plt.plot(np.arange(1980, 2015), cv_results['test']['predicted_yields'],
             marker='s', color='b', label='Predicted Yield')
    plt.plot(np.arange(1980, 2015), cv_results['test']['actual_yields'],
             marker='s', color='k', label='Observed Yield')
    plt.xlabel('Year')
    plt.ylabel('Yield (tonnes ha$^{-1}$)')
    plt.legend()
    plt.savefig('./figures/sliding_window_us_maize.pdf')
    plt.savefig('./figures/sliding_window_us_maize.png')
    plt.show()

    # fit = model.sampling(data, chains=args.chains, iter=args.iter,
    #                      verbose=args.verbose, seed=args.seed)
    # #
    # utils.plotting.compare_annual_mean_yield_prediction(fit, data, 'United States - Wheat (Winter)')
    # plt.show()
    # plt.savefig('./figures/Annual_Mean_Anom_Winter_Wheat_USA_6m')
    # plt.savefig('./figures/Annual_Mean_Anom_Winter_Wheat_USA_6m.png')

    # # print(fit)
    # ax = utils.plotting.plot_temp_precip_variation(fit, data)
    # ax.plot()
    # plt.savefig('./figures/Maize_Spring_USA_TP_6m.pdf')
    # plt.savefig('./figures/Maize_Spring_USA_TP_6m.png')
    # plt.show()
    # #
    # samples = fit.extract()
    # utils.plotting.plot_per_region_yield_predictions(fit, data, soybean_usa)
    # plt.show()
    # #
    # samples = fit.extract()
    # pm = model_utils.extract_parameter_means(samples)
    # ax = utils.plotting.plot_temp_precip_constant_6m((0, 40), (0, 250), pm['mu_t'], pm['mu_p'], pm['sigma_t'],
    #                                pm['sigma_p'], [pm['norm']], pm['rho'])
    # ax.plot()
    # plt.show()
