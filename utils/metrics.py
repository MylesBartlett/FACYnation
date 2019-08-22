from sklearn import metrics
import numpy as np


def rmse(y_pred, y_true):
    return np.sqrt(metrics.mean_squared_error(y_pred, y_true))


def rrmse(y_pred, y_true):
    norm_factor = 1 / np.ptp(y_true)
    return norm_factor * rmse(y_pred, y_true)


def nash_sutcliffe_eff(y_pred, y_true):
    return 1 - ((y_pred - y_true)**2).sum() / ((y_true - np.mean(y_true))**2).sum()
