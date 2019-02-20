import numpy as np
from utils import evaluate
import pylab


def _temp_precip_grid(temp_range, preip_range, mu_t, mu_p, sigma_t, sigma_p,
                      norm, rho, num_samples=100):
    x = np.linspace(*temp_range, num_samples)
    y = np.linspace(*preip_range, num_samples)
    X, Y = pylab.meshgrid(x, y)  # grid of point

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

    fig, ax = pylab.subplots()
    Z = _temp_precip_grid(temp_range, preip_range, mu_t, mu_p, sigma_t, sigma_p,
                          norm, rho, num_samples)

    im = ax.imshow(np.flip(Z, axis=0), cmap=cmap,
                   extent=temp_range+preip_range, aspect="auto")  # drawing the function

    cbar = fig.colorbar(im)  # adding the colobar on the right
    cbar.set_label('Yield [tonnes ha$^{-1}$]')

    ax.set_xlabel('T (Celsius)')
    ax.set_ylabel('Monthly precipitation (mm)')

    return ax
