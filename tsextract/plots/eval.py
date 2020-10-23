## Return cross correlation for Time Lag
import pandas as pd
import numpy as np
import scipy.stats

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()



def get_lag_corr(y_actual, y_pred, num_lags):
    lags = []
    for c in range(num_lags):
        lagged = pd.Series(y_pred).shift(c)
        lags.append(scipy.stats.spearmanr(lagged, y_actual, nan_policy='omit')[0])

    ax = sns.lineplot(range(len(lags)), lags, linestyle="-")
    ax.set(xlabel='Lag', ylabel='Corr Coefficient')


def actualPred(y_true, y_pred):
    ax = sns.lineplot(range(y_true.shape[0]), y_true,
                      color="blue", label="Actual", linestyle="-")

    ax = sns.lineplot(range(y_pred.shape[0]), y_pred,
                  color="yellow", label="Predicted", linestyle="-")

    ax.set(xlabel='Time', ylabel='Y')


def scatter(y_true, y_pred):
    ax = sns.scatterplot(y_true, y_pred)
    ax.set(xlabel='Actual', ylabel='Predicted')
