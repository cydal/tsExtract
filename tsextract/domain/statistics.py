import numpy as np
from scipy.stats import kurtosis, skew, moment, entropy, iqr, median_absolute_deviation

axis = 1

def mean(data):
    return np.mean(data, axis=axis)

def median(data):
    return np.median(data, axis=axis)

def std(data):
    return np.std(data, axis=axis)

def min(data):
    return np.ndarray.min(data, axis=axis)

def max(data):
    return np.ndarray.max(data, axis=axis)

def range(data):
    return np.ptp(data, axis=axis)

def variance(data):
    return np.var(data, axis=axis)

def kurtosis(data):
    return kurtosis(data, axis=axis)

def skew(data):
    return skew(data, axis=axis)

def iqr(data):
    return iqr(data, axis=axis)

def mae(data):
    return median_absolute_deviation(data, axis=axis)

def rms(data):
    return np.sqrt(np.mean(np.square(data), axis=axis))
