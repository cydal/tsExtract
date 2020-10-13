import numpy as np
from scipy.stats import kurtosis, skew, moment, entropy, iqr, median_absolute_deviation


class statistics:

    def __init__(self):
        self.axis = 1

    def mean(data):
        return np.mean(data, axis=self.axis)

    def median(data):
        return np.median(data, axis=self.axis)

    def std(data):
        return np.std(data, axis=self.axis)

    def min(data):
        return np.ndarray.min(data, axis=self.axis)

    def max(data):
        return np.ndarray.max(data, axis=self.axis)

    def range(data):
        return np.ptp(data, axis=self.axis)

    def variance(data):
        return np.var(data, axis=self.axis)

    def kurtosis(data):
        return kurtosis(data, axis=self.axis)

    def skew(data):
        return skew(data, axis=self.axis)

    def iqr(data):
        return iqr(data, axis=self.axis)

    def mae(data):
        return median_absolute_deviation(data, axis=self.axis)

    def rms(data):
        return np.sqrt(np.mean(np.square(data), axis=self.axis))
