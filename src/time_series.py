import pandas as pd
import numpy as np
from datetime import datetime
from pandas import concat
from scipy.stats import kurtosis, skew, moment, entropy, spearmanr
import seaborn as sns

from .utils import calc_window, check_cut, calc_difference, calc_momentum, calc_force, get_num_nan

# TODO:
# create method for user to request the combined dataset easily
# add methods to create more features?
# better comment code
# what are comb functions?
# autocorrelation, lag, etc?


class TimeSeries():
    """ Loads and transforms time series data
    TODO: complete description
    """

    def __init__(self, data):
        self.data = data

    def window_data(self, window):
        # before window_list
        """
        TODO all the descriptions
        Converts a datetime series to a wind
        Takes in 1-dim data, T+, window size
        data - 1 dim time series
        f_idx - T+ value /// leave @ 0 for the moment. may be irrelevant
        window - window - size
        """
        win_data = calc_window(self.data, window)

        return win_data

    # def build_data(self):
    #     # TODO: Through an input dictionary with all requests, and use
    #     # functions to construct the wanted dataset

    def window_stat(self, window, stat):
        """
        Returns 1-dim vector of summary stats
        Expects 2 dim matrix
        """
        win_data = self.window_data(window)

        if stat == "mean":
            return win_data.mean(axis=1)
        elif stat == "median":
            return np.median(win_data, axis=1)
        elif stat == "std":
            return win_data.std(axis=1)
        elif stat == "min":
            return win_data.min(axis=1)
        elif stat == "max":
            return win_data.max(axis=1)
        elif stat == "range":
            return np.ptp(win_data, axis=1)
        elif stat == "skew":
            return skew(win_data, axis=1)
        elif stat == "kurtosis":
            return kurtosis(win_data, fisher=False, axis=1)
        elif stat == "moment":
            return moment(win_data, moment=1, axis=1)
        return None

    def difference(self, d):
        dif_data = calc_difference(self.data, d)
        return dif_data

    def momentum(self, d):
        mom_data = calc_momentum(self.data, d)
        return mom_data

    def force(self, d):
        force_data = calc_force(self.data, d)
        return force_data

    def difference_comb(self, window, d, cutt=False):
        check_cut(window, d)

        win_data = pd.DataFrame(calc_window(self.data, window-1))
        multi_diff = calc_difference(win_data, d)
        single_diff = calc_difference(self.data, d)

        multi_size = multi_diff.shape[1]

        if cutt:
            multi_diff = multi_diff.iloc[:, d:]

        diffed = np.column_stack((multi_diff, single_diff.rename(multi_size)))

        return diffed

    def momentum_comb(self, window, d, cutt=False):
        check_cut(window, d)

        win_data = pd.DataFrame(calc_window(self.data, window-1))
        multi_diff = calc_momentum(win_data, d)
        single_diff = calc_momentum(self.data, d)

        multi_size = multi_diff.shape[1]

        if cutt:
            multi_diff = multi_diff.iloc[:, d:]

        diffed = np.column_stack((multi_diff, single_diff.rename(multi_size)))

        return diffed

    def force_comb(self, window, d, cutt=False):
        check_cut(window, d)

        win_data = pd.DataFrame(calc_window(self.data, window-1))
        multi_diff = calc_force(win_data, d)
        single_diff = calc_force(self.data, d)
        multi_size = multi_diff.shape[1]

        if cutt:
            multi_diff = multi_diff.iloc[:, d:]

        diffed = np.column_stack((multi_diff, single_diff.rename(multi_size)))

        return diffed
