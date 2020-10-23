import pandas as pd
import numpy as np
import scipy.stats

import sys
sys.path.append('../')

from scripts.helper import *


## Target Lag - T+ for predictive variable

## todo - write error checks if values differ from what is expected
#  arguments
#  window - integer // window_statistics - list, 2 values (int & string)
#  difference / momentum / force - list, 2 values (int & int)

## Returns 1-dim vector of summary stats
## Expects 2 dim matrix
def win_stat(data, fun_c):
  return(np.expand_dims(fun_c(data), axis=1))

## when building features, minimum required argument is windows size
## Automate feature name passed in to feature variable
# will allow to handle errors all at once

## implement disjointed backwards lag e.g t-24 to t-48 // t-48 to t-96 e.t.c


def build_features(data, features_request, target_lag=3, include_tzero=True):

  ## Check window option present
  if "window" not in features_request.keys():
    raise Exception("Features dictionary must contain at the least, a window size")

  # Check input format correct
  if len(features_request["window"]) != 1:
    if type(features_request["window"][0]) != int:
      raise Exception("Expects list of size 1 with integer value for window size")

  features_list = []
  features = {}

  # Helper functions to run different operations
  def window(features_item):
    return(window_list(data, features_item))

  def window_statistic(features_item):
    return(win_stat(window_list(data, features_item[:-1]), features_item[-1]))

  ## Confirm - :-1 --- -1
  def difference(features_item): ## 10 24 10
    return(difference_comb(data, features_item))

  def momentum(features_item):
    return(momentum_comb(data, features_item))

  def force(features_item):
    return(force_comb(data, features_item))

  ### Confirm  1-2 windowing
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

  # Save features to list
      ## Create list - first item is feature name
      ## & second item is data with op performed
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
      ### Windowed data - First & Always required
    if f[0] == "window":
    # Get window size for naming columns
    ## If single value window
      if len(features_request["window"]) == 1:
          window_range = ["T-{0}".format(i) for i in range(1, features_request["window"][0]+1)][::-1]
    ## If double value window - end & start - e.g every 12 hours e.t.c
      else:
          window_range = ["T-{0}".format(i) for i in range(features_request["window"][0]+1, features_request["window"][1]+1)][::-1]

      # Create DataFrame with window
      df = pd.DataFrame(f[1], columns=window_range)

  # Add feature name for stat features - window/lag/statfeature
  ## Pause
    if f[0] in stat_features: ## If stat feats
        feat_name = "_".join([str(x) for x in features_request[f[0]]])
        features[feat_name] = pd.DataFrame(f[1], columns=[feat_name])


  # Add multi row names for non-stat features
    if f[0] in nonstat_features: ## If non stat feats
      feat_name = "_".join([str(x) for x in features_request[f[0]]])

  # Add column names & save to dataframe
  ### Get the diff between window_size & lag value
  ## check size of input
      #win_lag_diff = features_request[f[0]][0] - features_request[f[0]][1]
      win_lag_diff = f[1].shape[1]
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
