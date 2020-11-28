## Return cross correlation for Time Lag
import pandas as pd
import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import plotnine
from plotnine import *

def get_lag_corr(y_actual, y_pred, num_lags):
    """Calculates & plots Lag Correlation

    Parameters
    ----------
    y_actual : Series or vector
        The ground truth
    y_pred : Series or vector
        Predicted values
    num_lags : int
        Lag to consider - range (0, num_lags)
    """
    lags = []
    for c in range(num_lags):
        lagged = pd.Series(y_pred).shift(c)
        lags.append(scipy.stats.spearmanr(lagged, y_actual, nan_policy='omit')[0])

    datum = pd.DataFrame({
            "Lags":range(len(lags)), 
            "Lag-Coefficient":lags
        })

    p = (
        ggplot(datum, aes(x='Lags'))
        + geom_line(aes(y='Lag-Coefficient'))
        + labs(x='Lag', y='Coefficient')
        + plotnine.theme_538()
        + plotnine.theme(figure_size=(10, 6))
    )
    print(p)


def actualPred(y_true, y_pred):
    """Plot actual vs predicted line plots

    Parameters
    ----------
    y_true : Series or vector
        The ground truth
    y_pred : Series or vector
        Predicted values
    """
    datum = pd.DataFrame({
        "date": range(y_true.shape[0]),
        "Actual":y_true, 
        "Prediction":y_pred
    })
    datum = pd.melt(datum, id_vars=['date'], value_vars=['Actual', 'Prediction']) 

    p = (
    ggplot(datum, aes(x='date'))
    + geom_line(aes(y='value', color='variable')) # line plot
    + labs(x='date', y='Solar Output')
    + plotnine.theme_538()
    + plotnine.theme(figure_size=(10, 6))
    )
    print(p)


def scatter(y_true, y_pred):
    """Plot actual vs predicted scatterplot

    Parameters
    ----------
    y_true : Series or vector
        The ground truth
    y_pred : Series or vector
        Predicted values
    """
    datum = pd.DataFrame({
        "Actual":y_true, 
        "Prediction":y_pred
    })

    p = (
    ggplot(datum, aes(x='Actual', y="Prediction", color='"#9B59B6"'))
    + geom_point() # line plot
    + labs(x='Actual', y='Prediction')
    + plotnine.theme_538()
    + plotnine.theme(figure_size=(10, 6))
    )
    print(p)
