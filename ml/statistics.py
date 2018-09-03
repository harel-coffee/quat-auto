#!/usr/bin/env python3
import sys
import numpy as np

import sklearn.metrics
import scipy.spatial


def _cohen_d(x, y):
    from numpy import std, mean, sqrt
    d = (mean(x) - mean(y)) / sqrt((std(x, ddof=1) ** 2 + std(y, ddof=1) ** 2) / 2.0)
    if (len(x) != len(y)):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (mean(x) - mean(y)) / sqrt(((nx - 1)*std(x, ddof=1) ** 2 + (ny - 1) * std(y, ddof=1) ** 2) / dof)
    return d


def rmse(data, X, Y):
    return np.sqrt(sklearn.metrics.mean_squared_error(data[X], data[Y]))


def cos_sim(x, y):
    try:
        d = scipy.spatial.distance.cosine(x, y)
        if np.isnan(d):
            return 0.0
        return d
    except:
        return 0.0


def r2(data, X, Y):
    return sklearn.metrics.r2_score(data[X], data[Y])


def cohen_d(data, X, Y):
    return _cohen_d(data[X], data[Y])


def calc_correlations(data, X, Y):
    """
    Take a data frame as input and calculate various correlations.

    Returns:
        dict with the following keys: pearson, kendall, spearman
    """
    correlation_values = {}
    for c in ['pearson', 'kendall', 'spearman']:
        correlation_values[c] = data[[X, Y]].corr(method=c)[Y][X]
    return correlation_values


def mean_absolute_error(data, X, Y):
    return sklearn.metrics.mean_absolute_error(data[X], data[Y])


def median_absolute_error(data, X, Y):
    return sklearn.metrics.median_absolute_error(data[X], data[Y])


def calc_regression_metrics(data, X, Y):
    result = calc_correlations(data, X, Y)
    result["r2"] = r2(data, X, Y)
    result["rmse"] = rmse(data, X, Y)
    result["cohen_d"] = cohen_d(data, X, Y)
    result["mean_absolute_error"] = mean_absolute_error(data, X, Y)
    result["median_absolute_error"] = median_absolute_error(data, X, Y)
    return result


if __name__ == "__main__":
    print("this is just a lib")
    sys.exit(0)
