import numpy as np


def hoeffding_anytime(t, delta, variance_proxy):
    return np.sqrt(2 * variance_proxy * np.log(4 * t ** 2 / delta) / t)


def update_mean_online(mean_support: np.ndarray, mean_hat: np.ndarray, new_val: np.ndarray):
    return (1 / (1 + mean_support)) * (mean_support * mean_hat + new_val)


def conf_interval(std, num_runs, quantile=1.96):
    return quantile * std / (num_runs ** 0.5)
