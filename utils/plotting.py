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


def plot_temp_precip_grid(temp_range, preip_range, mu_t, mu_p, sigma_t, sigma_p,
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
