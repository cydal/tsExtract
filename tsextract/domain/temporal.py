
import statsmodels.api as sm
import numpy as np



axis = 1

def abs_energy(data):
    return np.sum(data**2, axis=axis)

def area_under_curve(data):
    return np.trapz(data, axis=axis)

def mean_abs_diff(data):
    return(np.mean(np.abs(np.diff(data)), axis=axis))

def moment(data):
    return moment(data, axis=axis)

# Calculate autocorrelation function - default 2
def autocorrelate(data, lag=2):
    def _autoc(row, lag):
        return(sm.tsa.acf(row, nlags=lag, fft=False))

    return(np.apply_along_axis(_autoc, data, lag=lag)[:, lag])

## Calculate Zero Crossing rate
def zero_crossing(data):
    def _cross(row):
        return(np.where(np.diff(row > 0)))

    return(np.apply_along_axis(_cross, 1, data))
