import numpy as np
from scipy import stats
from scipy.stats import kurtosis, skew, moment, entropy, iqr, median_absolute_deviation

axis = 1

def mean(data):
    """Calculates mean

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    return np.mean(data, axis=axis)

def median(data):
    """Calculates median

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    return np.median(data, axis=axis)

def std(data):
    """Calculates Standard Deviation

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    return np.std(data, axis=axis)

def min(data):
    """Calculates minimum

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    return np.ndarray.min(data, axis=axis)

def max(data):
    """Calculates Maximum

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    return np.ndarray.max(data, axis=axis)

def range(data):
    """Calculates Range

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    return np.ptp(data, axis=axis)

def variance(data):
    """Calculates Variance

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    return np.var(data, axis=axis)

def kurtosis(data):
    """Calculates Kurtosis

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    return stats.kurtosis(data, axis=axis)

def skew(data):
    """Calculates Skew

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    return stats.skew(data, axis=axis)

def iqr(data):
    """Calculates IQR

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    return stats.iqr(data, axis=axis)

def mae(data):
    """Calculates MAE

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    return median_absolute_deviation(data, axis=axis)

def rmse(data):
    """Calculates RMSE

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    return np.sqrt(np.mean(np.square(data), axis=axis))
