#!/usr/bin/env python3
"""
Collection of statistics methods to evauluate machine learning classification and regression.
"""
"""
    This file is part of quat.
    quat is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    quat is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with quat. If not, see <http://www.gnu.org/licenses/>.

    Author: Steve Göring
"""
import sys
import numpy as np

import sklearn.metrics
import scipy.spatial


def _cohen_d(x, y):
    """ calculates cohen's d,
    based on https://stackoverflow.com/a/33002123

    Parameters
    ----------
    x : list/np.array
    y : list/np.array
    """
    from numpy import std, mean, sqrt
    d = (mean(x) - mean(y)) / sqrt((std(x, ddof=1) ** 2 + std(y, ddof=1) ** 2) / 2.0)
    if (len(x) != len(y)):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (mean(x) - mean(y)) / sqrt(((nx - 1)*std(x, ddof=1) ** 2 + (ny - 1) * std(y, ddof=1) ** 2) / dof)
    return d


def rmse(data, X, Y):
    """ rmse caluclation

    Parameters
    ----------
    data : pandas dataframe
        stores all data
    X : str
        X column to be used
    Y : str
        Y column to be used
    """
    assert(X in data and Y in data)
    return np.sqrt(sklearn.metrics.mean_squared_error(data[X], data[Y]))


def cos_sim(x, y):
    """ calculates cosine similartiy between vector x and vector y

    Parameters
    ----------
    x : list/np.array
    y : list/np.array
    """
    try:
        d = scipy.spatial.distance.cosine(x, y)
        if np.isnan(d):
            return 0.0
        return d
    except:
        return 0.0


def r2(data, X, Y):
    """ calculates r2 measure

    Parameters
    ----------
    data : pandas dataframe
        stores all data
    X : str
        X column to be used
    Y : str
        Y column to be used
    """
    assert(X in data and Y in data)
    return sklearn.metrics.r2_score(data[X], data[Y])


def cohen_d(data, X, Y):
    """ calculates cohen's d measure

    Parameters
    ----------
    data : pandas dataframe
        stores all data
    X : str
        X column to be used
    Y : str
        Y column to be used
    """
    assert(X in data and Y in data)
    return _cohen_d(data[X], data[Y])


def calc_correlations(data, X, Y):
    """
    Take a data frame as input and calculate 'pearson', 'kendall', 'spearman' correlations.

    Parameters
    ----------
    data : pandas dataframe
        stores all data
    X : str
        X column to be used
    Y : str
        Y column to be used

    Returns
    -------
    dictionary with the following keys: pearson, kendall, spearman
    """
    assert(X in data and Y in data)

    correlation_values = {}
    for c in ['pearson', 'kendall', 'spearman']:
        correlation_values[c] = data[[X, Y]].corr(method=c)[Y][X]
    return correlation_values


def mean_absolute_error(data, X, Y):
    """
    calculates mean absolute error

    Parameters
    ----------
    data : pandas dataframe
        stores all data
    X : str
        X column to be used
    Y : str
        Y column to be used
    """
    assert(X in data and Y in data)
    return sklearn.metrics.mean_absolute_error(data[X], data[Y])


def median_absolute_error(data, X, Y):
    """
    calculates median absolute error

    Parameters
    ----------
    data : pandas dataframe
        stores all data
    X : str
        X column to be used
    Y : str
        Y column to be used
    """
    assert(X in data and Y in data)
    return sklearn.metrics.median_absolute_error(data[X], data[Y])


def calc_regression_metrics(data, X, Y):
    """
    calculates several regression metrics: r2, rmse, cohen_d, mean_absolute_error, median_absolute_error

    Parameters
    ----------
    data : pandas dataframe
        stores all data
    X : str
        X column to be used
    Y : str
        Y column to be used

    Returns
    -------
    dictionary of all metric values
    """
    assert(X in data and Y in data)
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
