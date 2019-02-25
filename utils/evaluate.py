import numpy as np
from utils import distributions as dist


def yield_anomaly_6m_tp(temp, precip, mu_t, mu_p, sigma_t, sigma_p, norm, rho=None):
    """Take six months of T and P and return yield for given params.

    This should be identical to the function in the STAN model
    """
    n_months = len(temp)
    if len(norm) == 1:
        norm = norm * np.ones(n_months)
    dy = np.zeros(n_months)
    for month in range(n_months):
        dy[month] = norm[month] * dist.bivariate_normal(temp[month], precip[month],
                                                        mu_t, mu_p, sigma_t, sigma_p, rho)

    return np.sum(dy)


def compute_annual_yield_anom_6m_tp(data: dict, mu_t, mu_p, sigma_t, sigma_p,
                                    norm, rho=None, t_inc=0, p_inc=0, average=True):
    """
    Compute mean yield anomaly for a model over regions and years.
    If a correlation coefficient (rho) is provided, the correlation between
    temperature and precipitation will be taken into account.

    The function yield_anomaly returns the yield anomaly for a given year
    and region. Here we loop over the regions and years to create
    and overall mean.

    Args:
        data: dictionary of data
        mu_t: mean temperature
        sigma_t: standard deviation of the temperature
        mu_p: mean precipitation
        sigma_p: standard deviation of precipitation
        norm: normalization weights for each month
        rho: (Pearson) correlation coefficient. If given, correlation between
        the variables, temperature and precipitation, will be taken into account
        average: whether to average the yield anomalies across regions and years

    Returns:
        Annual yield anomalies
    """
    yield_anomalies = np.full((data['n_regions'], data['n_years']), np.nan)
    # loop over states
    for state in range(data['n_regions']):
        # loop over years
        for year in range(data['n_years']):
            temp = data['d_temp'][state, year, :] + t_inc
            precip = data['d_precip'][state, year, :] + p_inc
            yield_anomalies[state, year] = yield_anomaly_6m_tp(temp, precip, mu_t, mu_p,
                                                               sigma_t, sigma_p, norm, rho)
    if average:
        yield_anomalies = np.nanmean(yield_anomalies)

    return yield_anomalies
