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
  
  return(np.expand_dims(returned_lst, axis=1))


## Difference/Momentum/Force helper functions
def difference(data, lag):
  return(data.diff(lag))

def momentum(data, lag):
  return(difference(difference(data, lag), lag))

def force(data, lag):
  return(difference(momentum(data, lag), lag))



## Perform diff/mom/force operations
def difference_comb(data, window, lag):

## Number of features to return
  cut = window - lag

## Window size > Lag
  if cut <= 0:
    raise Exception("Window size must be greater than differencing")

  ## Difference Matrix
  multi_diff = difference(pd.DataFrame(window_list(data, window-1)), lag)
  single_diff = difference(data, lag)

  multi_size = multi_diff.shape[1]
  multi_diff = multi_diff.iloc[:, lag:]

  ## Matrix result from operation
  diffed = np.column_stack((multi_diff, single_diff.rename(multi_size)))

  return(diffed)

  #return(pd.DataFrame(diffed))

def momentum_comb(data, window, lag):

  ## Number of features to return
  cut = window - lag

  if cut <= 0:
    raise Exception("Window size must be greater than differencing")

  ## Momentum Matrix
  multi_diff = momentum(pd.DataFrame(window_list(data, window-1)), lag)
  single_diff = momentum(data, lag)

  multi_size = multi_diff.shape[1]
  multi_diff = multi_diff.iloc[:, lag:]

  ## Matrix result from operation
  diffed = np.column_stack((multi_diff, single_diff.rename(multi_size)))

  return(diffed)

def force_comb(data, window, lag):

  ## Number of features to return
  cut = window - lag

  if cut <= 0:
    raise Exception("Window size must be greater than differencing")

  ## Force Matrix
  multi_diff = force(pd.DataFrame(window_list(data, window-1)), lag)
  single_diff = force(data, lag)

  multi_size = multi_diff.shape[1]
  multi_diff = multi_diff.iloc[:, lag:]

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

## Return cross correlation for Time Lag

def get_lag_corr(pred, actual, num_lags):
    lags = []
    for c in range(num_lags):
        lagged = pd.Series(pred).shift(c)
        lags.append(scipy.stats.spearmanr(lagged, actual, nan_policy='omit')[0])
        
    return(lags)



## Target Lag - T+ for predictive variable

## todo - write error checks if values differ from what is expected
#  arguments
#  window - integer // window_statistics - list, 2 values (int & string)
#  difference / momentum / force - list, 2 values (int & int)


## when building features, minimum required argument is windows size - DONE
## Automate feature name passed in to feature variable - DONE
#  will allow to handle errors all at once

## implement disjointed backwards lag e.g t-24 to t-48 // t-48 to t-96 e.t.c


def build_features(data, features_request, target_lag=3, include_tzero=True):

  ## Check window option present
  if "window" not in features_request.keys():
    raise Exception("Features dictionary must contain at least a window size")

  # Check input format correct
  if len(features_request["window"]) != 1:
    if type(features_request["window"][0]) != int:
      raise Exception("Expects list of size 1 with integer value for window size")

  features_list = []
  features = {}

  # Helper functions to run different operations
  def window(features_item):
    return(window_list(data, features_item[0]))

  def window_statistic(features_item):
    return(win_stat(window_list(data, features_item[0]), features_item[1]))

  def difference(features_item):
    return(difference_comb(data, features_item[0], features_item[1]))

  def momentum(features_item):
    return(momentum_comb(data, features_item[0], features_item[1]))

  def force(features_item):
    return(force_comb(data, features_item[0], features_item[1]))

  def difference_statistic(features_item):
    return(win_stat(difference_comb(data, features_item[0], features_item[1]), 
             features_item[2]))

  def momentum_statistic(features_item):
    return(win_stat(momentum_comb(data, features_item[0], features_item[1]), 
             features_item[2]))
    
  def force_statistic(features_item):
    return(win_stat(force_comb(data, features_item[0], features_item[1]), 
             features_item[2]))

  ## Store feature names of type statistic
  stat_features = ["window_statistic", "difference_statistic", 
                   "momentum_statistic", "force_statistic"]

  ## Store feature names of non statistic type
  nonstat_features = ["difference", "momentum", "force"]

  ## Save features to list
  for key in features_request:
    features_list.append([key, locals()[key](features_request[key])])

  ## Get T+0 if Tplus0 is true
  if include_tzero:
    tzero = pd.DataFrame(np.expand_dims(data, axis=1), columns=["tzero"])

  # Get T+ target
  target = pd.DataFrame(np.expand_dims(data.shift(-target_lag), axis=1), 
                        columns=["Target_Tplus{0}".format(str(target_lag))])

  ### Loop through features
  ## Ensure window data is processed first
  for f in features_list:
    if f[0] == "window":
      # Get window size for naming columns
      window_range = ["T-{0}".format(i) for i in range(1, features_request["window"][0]+1)][::-1]
      # Create DataFrame with window
      df = pd.DataFrame(f[1], columns=window_range)
  # Add feature name for stat features - window/lag/statfeature
    if f[0] in stat_features:
      feat_name = "_".join([str(x) for x in features_request[f[0]]])
      features[feat_name] = pd.DataFrame(f[1], columns=[feat_name])
  # Add multi row names for non-stat features
    if f[0] in nonstat_features:
      feat_name = "_".join([str(x) for x in features_request[f[0]]])

  # Add column names & save to dataframe
      win_lag_diff = features_request[f[0]][0] - features_request[f[0]][1]
      window_range = ["{1}-{0}".format(i, feat_name) for i in range(1, win_lag_diff+1)][::-1]
      features[feat_name] = pd.DataFrame(f[1], columns=window_range)

  
  # Combine rest of features
  for d in features:
    df = pd.concat([df, features[d]], axis=1)

  # Add t+1 & t+0
  if include_tzero:
    df = pd.concat([df, tzero], axis=1)

  ## Add target column
  df = pd.concat([df, target], axis=1)

  # Set index of original dataframe
  df["Date"] = data.index
  df = df.set_index("Date")

  # Remove NaN values
  return(cut_final(df))