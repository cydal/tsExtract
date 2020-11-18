import pandas as pd
import numpy as np
from datetime import datetime
from pandas import concat



## Description takes in 1-dim data, T+, window size
### data - 1 dim time series
### f_idx - T+ value /// leave @ 0 for the moment. may be irrelevant
### window - window - size

def window_list(data, window):
  if len(window) == 1:
    return(np.vstack([data.shift(i + 1) for i in range(window[0])][::-1]).T)
  else:
    return(np.vstack([data.shift(i + 1) for i in range(window[0], window[1])][::-1]).T)


## Current - Window list - (data, window - int)
## Ideal - window list - (data, [int] / [int, int])

# Feature_list -> (dict_object) window() -> window_list
# Feature_list -> (dict_object) window_stat() -> window_list
# Feature_list -> (dict_object) difference_comb() -> window_list



## Difference/Momentum/Force helper functions
def difference(data, lag):
  return(data.diff(lag))

def momentum(data, lag):
  return(difference(difference(data, lag), lag))

def force(data, lag):
  return(difference(momentum(data, lag), lag))



## Perform diff/mom/force operations
def difference_comb(data, window_lag):
  ## Check size of window_size
  if len(window_lag) == 2:
    ## Number of features to return
    cut = window_lag[0] - window_lag[1]
  else:
    cut = window_lag[1] - window_lag[2]

  ## Window size > Lag
  if cut <= 0:
    raise Exception("Window size must be greater than differencing")

  if len(window_lag) == 2:
    ## Difference Matrix
    multi_diff = difference(pd.DataFrame(window_list(data, [window_lag[0]-1])), window_lag[1])
    single_diff = difference(data, window_lag[1])

    multi_size = multi_diff.shape[1]
    multi_diff = multi_diff.iloc[:, window_lag[1]:]
  else:
    ## Difference Matrix
    multi_diff = difference(pd.DataFrame(window_list(data,[window_lag[0]-1 , window_lag[1]-1])), window_lag[2])
    single_diff = difference(data, window_lag[2])

    multi_size = multi_diff.shape[1]
    multi_diff = multi_diff.iloc[:, window_lag[2]:]

  ## Matrix result from operation
  diffed = np.column_stack((multi_diff, single_diff.rename(multi_size)))

  return(diffed)

  #return(pd.DataFrame(diffed))

def momentum_comb(data, window_lag):

  ## Check size of window_size
  if len(window_lag) == 2:
    ## Number of features to return
    cut = window_lag[0] - window_lag[1]
  else:
    cut = window_lag[1] - window_lag[2]


  if cut <= 0:
    raise Exception("Window size must be greater than differencing")

  if len(window_lag) == 2:
    ## Momentum Matrix
    multi_diff = momentum(pd.DataFrame(window_list(data, [window_lag[0]-1])), window_lag[1])
    single_diff = momentum(data, window_lag[1])

    multi_size = multi_diff.shape[1]
    multi_diff = multi_diff.iloc[:, window_lag[1]:]
  else:
    ## Momentum Matrix
    multi_diff = momentum(pd.DataFrame(window_list(data,[window_lag[0]-1 , window_lag[1]-1])), window_lag[2])
    single_diff = momentum(data, window_lag[2])

    multi_size = multi_diff.shape[1]
    multi_diff = multi_diff.iloc[:, window_lag[2]:]

  ## Matrix result from operation
  diffed = np.column_stack((multi_diff, single_diff.rename(multi_size)))

  return(diffed)

def force_comb(data, window_lag):

  ## Check size of window_size
  if len(window_lag) == 2:
  ## Number of features to return
    cut = window_lag[0] - window_lag[1]
  else:
    cut = window_lag[1] - window_lag[2]

  if cut <= 0:
    raise Exception("Window size must be greater than differencing")

  if len(window_lag) == 2:
    ## Force Matrix
    multi_diff = difference(pd.DataFrame(window_list(data, [window_lag[0]-1])), window_lag[1])
    single_diff = difference(data, window_lag[1])

    multi_size = multi_diff.shape[1]
    multi_diff = multi_diff.iloc[:, window_lag[1]:]
  else:
    ## Force Matrix
    multi_diff = difference(pd.DataFrame(window_list(data,[window_lag[0]-1 , window_lag[1]-1])), window_lag[2])
    single_diff = difference(data, window_lag[2])

    multi_size = multi_diff.shape[1]
    multi_diff = multi_diff.iloc[:, window_lag[2]:]

  ## Matrix result from operation
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
