import numpy as np
import utils
import pylab as plt
import pandas as pd


def _temp_precip_grid(temp_range, preip_range, mu_t, mu_p, sigma_t, sigma_p,
                      norm, rho, num_samples=100):
    x = np.linspace(*temp_range, num_samples)
    y = np.linspace(*preip_range, num_samples)
    X, Y = plt.meshgrid(x, y)  # grid of point

    def z_func(X, Y):
        temp_6m = np.full(6, X)
        precip_6m = np.full(6, Y)
        return utils.evaluate.yield_anomaly_tp(temp_6m, precip_6m, mu_t, mu_p,
                                               sigma_t, sigma_p, norm, rho)

    z_func = np.vectorize(z_func)
    Z = z_func(X, Y)  # evaluation of the function on the grid
    return Z


def plot_temp_precip_constant_6m(temp_range, preip_range, mu_t, mu_p, sigma_t, sigma_p,
                                 norm, rho, num_samples=100, cmap='RdBu'):
    fig, ax = plt.subplots()
    Z = _temp_precip_grid(temp_range, preip_range, mu_t, mu_p, sigma_t, sigma_p,
                          norm, rho, num_samples)

    im = ax.imshow(np.flip(Z, axis=0), cmap=cmap,
                   extent=temp_range + preip_range, aspect="auto")  # drawing the function

    cbar = fig.colorbar(im)  # adding the colobar on the right
    cbar.set_label('Yield [tonnes ha$^{-1}$]')

    ax.set_xlabel('T (Celsius)')
    ax.set_ylabel('Monthly precipitation (mm)')

    return ax


def plot_per_region_yield_predictions(fit, data, regions):
    samples = fit.extract()
    plt.figure(figsize=(10, 5 * len(regions)))
    for s in range(0, len(regions)):
        plt.subplot(len(regions), 1, s + 1)
        plt.violinplot(samples['pred_yields'][:, s, :], showextrema=False)
        plt.xticks(range(1, data['n_years']+1), np.arange(1980, 1980+data['n_years']-1), rotation=90)
        plt.plot(range(1, data['n_years']+1), fit.data['d_yields'][s, :])
        plt.title(regions[s])


def plot_temp_precip_variation(fit, data, save_path=''):
    T_inc = np.linspace(-10, 10, 50)
    P_inc = np.linspace(-100, 100, 50)

    samples = fit.extract()

    take_100 = np.random.choice([0,1],
                                size=len(samples['mu_t']),
                                p=(1-100/len(samples['mu_t']), 100/len(samples['mu_t'])))
    mean_yield_anom = np.full((len(T_inc), len(P_inc)), np.nan)

    for n, t in enumerate(T_inc):
        print("{} out of {}".format(n, len(T_inc)))
        for m, p in enumerate(P_inc):
            mean_yield_samples = np.full(len(samples['mu_t']), np.nan)
            for k in np.arange(len(samples['mu_t'])):
                if not take_100[k]:
                    continue

                # print('k = {}'.format(k))
                mu_t = samples['mu_t'][k]
                sigma_t = samples['sigma_t'][k]
                mu_p = samples['mu_p'][k]
                sigma_p = samples['sigma_p'][k]
                rho = samples['rho'][k] if 'rho' in samples else None
                norm = [samples['norm'][k]]

                mean_yield_samples[k] = utils.evaluate.compute_annual_yield_anom_tp(data, mu_t, mu_p, sigma_t,
                                                                                    sigma_p, norm, rho, t_inc=t, p_inc=p)

            mean_yield_anom[n, m] = np.nanmean(mean_yield_samples)

    fig, ax = plt.subplots()

    im = plt.imshow(np.flip(mean_yield_anom.T, axis=0), cmap='RdBu',
                    extent=[T_inc[0], T_inc[-1], P_inc[0], P_inc[-1]],
                    aspect="auto")  # drawing the function
    cbar = fig.colorbar(im)  # adding the colorbar on the right
    cbar.set_label('Yield [tonnes ha$^{-1}$]')
    ax.set_xlabel('$T_{inc}$ [K]')
    ax.set_ylabel('$P_{inc}$ [mm]')
    if save_path != '':
        plt.savefig(save_path)

    return ax


def plot_anomaly_trend(crop, season, country, region='Indiana'):
    key = [crop, season, country, region] if season != ''\
        else [crop, country, region]
    key = '_'.join(key)

    anom_yields = pd.read_table(f'../Crop_data_files/{crop}_median_yield_anoms.csv')
    real_yields = pd.read_csv(f'../Crop_data_files/{crop}_yield_obs_timeseries.csv')

    anom_yields = anom_yields.loc[anom_yields['Region'] == key]
    real_yields = real_yields.loc[real_yields['Region'] == key]

    # print(anom_yields.columns)
    anom_yields = anom_yields.iloc[:, 2:].values.squeeze()
    # print(real_yields.columns)
    real_yields = real_yields.iloc[:, 1:].values.squeeze()

    x = np.arange(1960, 2015, 1)
    plt.plot(x, real_yields, label='Yield', color='k', marker='s')
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, real_yields, 1))(np.unique(x)),
             label='Trend', alpha=0.5, color='k', ls='--')
    plt.plot(x, anom_yields, color='b', marker='s', label='Anomalies (Detrended Yield')
    plt.xlabel('Year')
    plt.ylabel(r'Maize yield (tons ha$^{-1}$)')
    plt.xticks(np.arange(1960, 2015, 5))
    plt.title('Indiana')
    plt.legend()
    plt.show()

    return anom_yields, real_yields


def compare_annual_mean_yield_prediction(fit, data, title: str):
    samples = fit.extract()
    pm = utils.model_utils.extract_parameter_means(samples)
    y_pred = utils.evaluate.compute_annual_yield_anom_tp(data, pm['mu_t'], pm['mu_p'], pm['sigma_t'],
                                                         pm['sigma_p'], [pm['norm']], pm['rho'], average=False)
    y_pred_mean_over_regions = np.mean(y_pred, axis=0)
    y_true_mean_over_regions = np.mean(data['d_yields'], axis=0)
    years = 1980 + np.arange(data['n_years'])
    plt.plot(years, y_true_mean_over_regions, label='Observation', lw=2, c='k', marker='s')
    plt.plot(years, y_pred_mean_over_regions, label='Hindcast', lw=2, c='b', marker='s')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Yield anomaly')
    plt.legend()
