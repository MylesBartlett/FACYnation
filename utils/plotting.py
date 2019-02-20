import numpy as np
from utils import evaluate
import pylab as plt


def _temp_precip_grid(temp_range, preip_range, mu_t, mu_p, sigma_t, sigma_p,
                      norm, rho, num_samples=100):
    x = np.linspace(*temp_range, num_samples)
    y = np.linspace(*preip_range, num_samples)
    X, Y = plt.meshgrid(x, y)  # grid of point

    def z_func(X, Y):
        temp_6m = np.full(6, X)
        precip_6m = np.full(6, Y)
        return evaluate.yield_anomaly_6m_tp(temp_6m, precip_6m, mu_t, mu_p,
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


def plot_per_region_yield_predictions(fit, regions):
    samples = fit.extract()
    plt.figure(figsize=(10, 5 * len(regions)))
    for s in range(0, len(regions)):
        plt.subplot(len(regions), 1, s + 1)
        plt.violinplot(samples['pred_yields'][:, s, :], showextrema=False)
        plt.xticks(range(1, 36), np.arange(1980, 2015), rotation=90)
        plt.plot(range(1, 36), fit.data['d_yields'][s, :])
        plt.title(regions[s])


def plot_temp_precip_variation(fit, data):
    T_incs_2 = np.linspace(-10, 10, 50)
    P_incs_2 = np.linspace(-100, 100, 50)

    samples = fit.extract()

    take_100 = np.random.choice([0,1],
                                size=len(samples['mu_t']),
                                p=(1-100/len(samples['mu_t']), 100/len(samples['mu_t'])))
    mean_yield_anom = np.full((len(T_incs_2), len(P_incs_2)), np.nan)

    for n, t in enumerate(T_incs_2):
        print("{} out of {}".format(n, len(T_incs_2)))
        for m, p in enumerate(P_incs_2):
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

                mean_yield_samples[k] = evaluate.compute_annual_yield_anom_6m_tp(data, mu_t, mu_p, sigma_t,
                                                                                 sigma_p, norm, rho, t, p)

            mean_yield_anom[n, m] = np.nanmean(mean_yield_samples)

    fig, ax = plt.subplots()

    im = plt.imshow(np.flip(mean_yield_anom.T, axis=0), cmap='RdBu',
                    extent=[T_incs_2[0], T_incs_2[-1], P_incs_2[0], P_incs_2[-1]],
                    aspect="auto")  # drawing the function
    cbar = fig.colorbar(im)  # adding the colorbar on the right
    cbar.set_label('Yield [tonnes ha$^{-1}$]')
    ax.set_xlabel('$T_{inc}$ [K]')
    ax.set_ylabel('$P_{inc}$ [mm]')

    return ax