import numpy as np


def bivariate_normal(x, y, mean_x, mean_y, std_x, std_y, corr_coeff=None):
    """
    (unnormalized) PDF of a bivariate normal distribution with factoring in correlation
    between random variables, x and y.
    Args:
        x: first correlated random variable
        y: second correlated random variable
        mean_x: mean of the first random variable,x
        mean_y: mean of the second random variable, y
        std_x: standard deviation of the first random variable, x
        std_y: standard deviation of the second random variable, y
        corr_coeff: Pearson correlation coefficient describing the correlation
        between random variables x and y. Given a pair of random variables, correlation
        coefficient is given as: Cov(X, Y) / (Var(X)Var(Y))

    Returns:
        probability of random variables x and y under a bivariate normal distribution
        with correlation between its variables

    """
    prefactor = -0.5

    diff_x = (x - mean_x) / std_x
    diff_y = (y - mean_y) / std_y

    exp = diff_x**2 + diff_y**2

    if corr_coeff is not None:
        prefactor /= 1 - corr_coeff**2
        exp -= (2 * corr_coeff * diff_x * diff_y / (std_x * std_y))

    return np.exp(prefactor * exp)
