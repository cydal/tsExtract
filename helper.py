import pandas as pd
import numpy as np
from datetime import datetime
from pandas import concat

from scipy.stats import kurtosis, skew, moment, entropy
import seaborn as sns



## Description takes in 1-dim data, T+, window size
### data - 1 dim time series
### f_idx - T+ value /// leave @ 0 for the moment. may be irrelevant
### window - window - size

def window_list(data, window):
  return(np.vstack([data.shift(i + 1) for i in range(window)][::-1]).T)



## Returns 1-dim vector of summary stats
## Expects 2 dim matrix

def win_stat(wnd_list, stat):
  if stat == "mean":
    returned_lst = wnd_list.mean(axis=1)
  elif stat == "median":
    returned_lst = np.median(wnd_list, axis=1)
  elif stat == "std":
    returned_lst = wnd_list.std(axis=1)
  elif stat == "min":
    returned_lst = wnd_list.min(axis=1)
  elif stat == "max":
    returned_lst = wnd_list.max(axis=1)
  elif stat == "range":
    returned_lst = np.ptp(wnd_list, axis=1)
  elif stat == "skew":
    returned_lst = skew(wnd_list, axis=1)
  elif stat == "kurtosis":
    returned_lst = kurtosis(wnd_list, fisher=False, axis=1)
  elif stat == "moment":
    returned_lst = moment(wnd_list, moment=1, axis=1)

  else:
    returned_lst = None
  
  return(returned_lst)


def difference(data, d):
  return(data.diff(d))

def momentum(data, d):
  return(difference(difference(data, d), d))

def force(data, d):
  return(difference(momentum(data, d), d))




def difference_comb(data, window, d, cutt=False):

  cut = window - d

  if cut <= 0:
    raise Exception("Window size must be greater than differencing")

  multi_diff = difference(pd.DataFrame(window_list(data, window-1)), d)
  single_diff = difference(data, d)

  multi_size = multi_diff.shape[1]

  if cutt:
    multi_diff = multi_diff.iloc[:, d:]

  diffed = np.column_stack((multi_diff, single_diff.rename(multi_size)))

  return(diffed)

  #return(pd.DataFrame(diffed))

def momentum_comb(data, window, d, cutt=False):

  cut = window - d

  if cut <= 0:
    raise Exception("Window size must be greater than differencing")

  multi_diff = momentum(pd.DataFrame(window_list(data, window-1)), d)
  single_diff = momentum(data, d)

  multi_size = multi_diff.shape[1]

  if cutt:
    multi_diff = multi_diff.iloc[:, d:]

  diffed = np.column_stack((multi_diff, single_diff.rename(multi_size)))

  return(diffed)

def force_comb(data, window, d, cutt=False):

  cut = window - d

  if cut <= 0:
    raise Exception("Window size must be greater than differencing")

  multi_diff = force(pd.DataFrame(window_list(data, window-1)), d)
  single_diff = force(data, d)

  multi_size = multi_diff.shape[1]

  if cutt:
    multi_diff = multi_diff.iloc[:, d:]


  diffed = np.column_stack((multi_diff, single_diff.rename(multi_size)))

  return(diffed)




### Get number of NaN values to drop
### returns NaNs at start & bottom of last T+ column.

def get_num_nan(df):
  num_nan = df.isnull().sum()
  return([max(num_nan[0:-1]), list(num_nan)[-1]])



## 2-d Matrix to Pandas Dataframe
## adds global index as DF index
## adds convenience for combination later

def build_data(data, index):
  data_df = pd.DataFrame(data)
  data_df["Date"] = index
  data_df = data_df.set_index("Date")
  return(data_df)




## Re-Centers data after multiple shifts
## uses num_nan output
def cut_final(df):
  h, t = get_num_nan(df)

  return(df[h:][:-t])

## Return cross correlation for Time Lag

def get_lag_corr(pred, actual, num_lags):
    lags = []
    for c in range(num_lags):
        lagged = pd.Series(pred).shift(c)
        lags.append(scipy.stats.spearmanr(lagged, actual, nan_policy='omit')[0])
        
    return(lags)