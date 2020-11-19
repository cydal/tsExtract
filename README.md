![logo](https://i.postimg.cc/rsVLMjzn/tsextract-logo.jpg)


## tsExtract: Time Series Preprocessing Library

tsExtract is a time series preprocessing library. Using sliding windows, tsExtract allows for the conversion of time series data to a form that can be fed into standard machine learning regression algorithms like Linear Regression, Decision Trees Regression and as well as Deep Learning. 

![enter image description here](https://img.shields.io/badge/LICENSE-GNU_GPL-BLACK) ![enter image description here](https://img.shields.io/badge/pypi-v1.0.0-yellow)  ![enter image description here](https://anaconda.org/cydal/tsextract/badges/version.svg)  ![enter image description here](https://anaconda.org/cydal/tsextract/badges/latest_release_date.svg)  ![enter image description here](https://anaconda.org/cydal/tsextract/badges/platforms.svg)    ![enter image description here](https://anaconda.org/cydal/tsextract/badges/installer/conda.svg)  
[![Open Source Helpers](https://www.codetriage.com/cydal/tsextract/badges/users.svg)](https://www.codetriage.com/cydal/tsextract)



# Installation

<code> pip </code>

> **pip install tsextract**

<code> conda </code>
> **conda install -c cydal tsextract**



## Main Features

* Take sliding window of data and with that, create additional columns representing the window. 
* Perform differencing on windowed data to remove non-stationarity. 
* Calculate statistics on windowed and differenced data. These include temporal and spectral statistics functions. 
* Plot visualisations. These include - 
* * Actual vs Predicted line and scatter plots
* * Lag correlation

## Usage

### [Example Notebooks](https://github.com/cydal/tsExtract/tree/master/examples)

```python
print(df.head())
```


|                |Date                          |DAYTON_MW                         |
|----------------|-------------------------------|-----------------------------|
| |`2004-12-31 01:00:00`            |`1596.0`            |
|          |`2004-12-31 02:00:00` | `1517.0` |
|          |`2004-12-31 03:00:00`|`1486.0`|
| | `2004-12-31 04:00:00`|`1469.0` |
| |`2004-12-31 05:00:00` | `1472.0` |



Using the main **build_features** function


**build_features** takes in 4 arguments - 
* **Data**: Time series data in 1d. 

* **Request Dictionary**: Dictionary with the function type and parameters
* **Include_tzero** (optional) - This gives the option on whether to include the column t+0. Can be quite handy when implementing difference networks. 
* **target_lag** - Sets lag value. If predicting 10 hours into the future, then a value of 10 should be included. Default is 3. 

```python
from tsextract.feature_extraction.extract import build_features

features_request = {
    "window":[10]
}

features = build_features(df["DAYTON_MW"], features_request, include_tzero=False)
```

The example above sends in a request for a sliding window size of 10. What is returned is a dataframe with 10 columns equal to the window size passed in. The final column is the target column with values shifted 3 time steps in the future. 


![enter image description here](https://i.postimg.cc/SRQTtbnH/Screenshot-2020-11-11-at-00-12-11.png)


### Features

* **window**: Takes sliding window of the data. Parameter(s) passed in as a list. A single value will take a sliding window corresponding to that value. A parameter of 10 will take windows from 1 to 10. If [5, 10] is passed in instead, then a window of 5 to 10 time steps will be taken instead. 

* **window_statistic**: This performs windowing like above, but then applies specified statistic operation to reduce matrix to a vector of 1d. 

* **difference/momentum/force**: Performs differencing by subtracting from the value in the present time step, the value in the previous time step. The parameter expected is a list of size 2 or 3. Just like in windowing, the first value refers to the window size. Two windowing values may also be passed in for windows in that range. 
The final value is the lag, this refers to the differencing lag for subtraction. A difference lag of 1 means values are subtracted from immediate past values (t3-t2, t2-t1, t1-t0 e.t.c) while a difference lag of 3 will subtract from 3 time steps before (t6-t3, t5-t2, t4-t1 e.t.c).
Momentum & Force are 2nd & 3rd order differences. 

* **difference_statistic/momentum_statistic/force_statistic**: Similarly, this performs the operations described above, but then applies the specified statistic. 

```python
from tsextract.feature_extraction.extract import build_features
from tsextract.domain.statistics import median, mean, skew, kurtosis
from tsextract.domain.temporal import abs_energy

features_request = {
    "window":[2], 
    "window_statistic":[24, median], 
    "difference":[12, 10],
    "difference_statistic":[15, 10, abs_energy], 
}

features = build_features(df["DAYTON_MW"], features_request, include_tzero=True, target_lag=3)
```

![enter image description here](https://i.postimg.cc/VvVhrsgm/Screenshot-2020-11-11-at-01-00-16.png)

# Summary Statistics


As described above, rather than take raw windowing or differencing matrix values, it is possible to take some summary statistic of it. See supported features. 


| Statistics      | Temporal | Spectral   |
| :---        |    :----:   |          ---: |
| Mean      | Absolute Energy       | Spectral Centroid   |
| Median   | AUC        |      |
| Range   | Mean Absolute Difference        |       |
| Standard Deviation   | Moment        |      |
| Minimum   | Autocorrelation        |     |
| Maximum   | Zero Crossing Rate         |   |
| Range   |         |      |
| Variance   |         |     |
| Kurtosis   |         |    |
| Skew   |         |     |
| IQR   |         |     |
| MAE   |         |     |
| RMSE   |         |     |




## Dependencies

* pandas >= 1.0.3
* seaborn >= 0.10.1
* statsmodels >= 0.11.1
* scipy >= 1.5.0
* matplotlib >= 3.2.1
* numpy >= 1.16.4


## License

[GNU GPL V3](http://www.gnu.org/licenses/quick-guide-gplv3.html)


# Contribute

Contributors of all experience levels are welcome. Please see the contributing guide. 


### Source Code

<code> You can get the latest source code </code>

> git clone https://github.com/cydal/tsExtract.git 
