
import statsmodels.api as sm
import numpy as np



axis = 1

def abs_energy(data):
    """Calculates Absolute Energy

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    return np.sum(data**2, axis=axis)

def area_under_curve(data):
    """Calculates AUC

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    return np.trapz(data, axis=axis)

def mean_abs_diff(data):
    """Calculates Mean Absolute Difference

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    return(np.mean(np.abs(np.diff(data)), axis=axis))

def moment(data):
    """Calculates Moment

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    return moment(data, axis=axis)

# Calculate autocorrelation function - default 2
def autocorrelate(data, lag=2):
    """Calculates AutoCorrelation

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on
    lag : int
        AutoCorrelation Lag

    Returns
    -------
    Array
        1-dim array
    """
    def _autoc(row, lag):
        return(sm.tsa.acf(row, nlags=lag, fft=False))

    return(np.apply_along_axis(_autoc, data, lag=lag)[:, lag])

## Calculate Zero Crossing rate
def zero_crossing(data):
    """Calculates Zero Crossing Rate

    Parameters
    ----------

    data : Matrix or DataFrame
        Data to perform operation on

    Returns
    -------
    Array
        1-dim array
    """
    def _cross(row):
        return(np.where(np.diff(row > 0)))

    return(np.apply_along_axis(_cross, 1, data))
