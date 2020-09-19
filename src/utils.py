import numpy as np
import pandas as pd
import scipy

# time_series helper functions


def calc_window(data, window):
    return np.vstack([data.shift(i + 1)
                      for i in range(window)][::-1]).T


def check_cut(window, d):
    """
    Check window size and differencing
    """
    cut = window - d
    if cut <= 0:
        raise Exception("Window size must be greater than differencing")


def calc_difference(data, d):
    dif_data = data.diff(d)
    return dif_data


def calc_momentum(data, d):
    mom_data = calc_difference(calc_difference(data, d), d)
    return mom_data


def calc_force(data, d):
    force_data = calc_difference(calc_momentum(data, d), d)
    return force_data


# functions to create final dataset


def get_num_nan(df):
    """
    Get number of NaN values to drop,
    returns NaNs at start & bottom of last T + column
    """
    num_nan = df.isnull().sum()
    return([max(num_nan[0:-1]), list(num_nan)[-1]])


def cut_final(df):
    """
    Re-Centers data after multiple shifts
    uses num_nan output
    """
    h, t = get_num_nan(df)
    cut = df[h:][:-t]
    return cut

# other


def build_data(data, index):
    """
    2-d Matrix to Pandas Dataframe
    adds global index as DF index
    adds convenience for combination later
    """
    data_df = pd.DataFrame(data)
    data_df["Date"] = index
    data_df = data_df.set_index("Date")
    return data_df


def get_lag_corr(pred, actual, num_lags):
    """
    Return cross correlation for Time Lag
    """
    lags = []
    for c in range(num_lags):
        lagged = pd.Series(pred).shift(c)
        lags.append(scipy.stats.spearmanr(
            lagged, actual, nan_policy='omit')[0])
    return lags
