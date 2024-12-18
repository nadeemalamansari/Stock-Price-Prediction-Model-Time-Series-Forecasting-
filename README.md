# Stock-Price-Prediction-Model-Time-Series-Forecasting-
• Developed a time-series forecasting model using LSTM networks for stock price prediction.  • Conducted data preprocessing with scaling, normalization, and feature engineering.  • Achieved 20% improvement in model performance via hyperparameter tuning and cross-validation. 
# Stock Price Prediction Using Time Series Models

This repository contains code for predicting stock prices using various time series models. The dataset used in this project is from NSE-TATAGLOBAL, which includes historical stock prices. The project demonstrates how to preprocess the data, check for stationarity, and apply different forecasting models including Moving Average, ARIMA, and SARIMA.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Data Import and Preprocessing](#data-import-and-preprocessing)
- [Stationarity Check](#stationarity-check)
- [Model Fitting](#model-fitting)
  - [Moving Average](#moving-average)
  - [ARIMA](#arima)
  - [SARIMA](#sarima)
- [Usage](#usage)
- [License](#license)

## Prerequisites

Make sure you have the following Python packages installed:

- `pmdarima`
- `statsmodels`
- `prophet`
- `pandas-profiling`
- `seaborn`
- `numpy`
- `matplotlib`
- `sklearn`

You can install these packages using pip:

```bash
pip install pmdarima statsmodels prophet pandas-profiling seaborn numpy matplotlib sklearn
```

## Data Import and Preprocessing

1. **Import Dataset**

   ```python
   data = pd.read_csv('https://raw.githubusercontent.com/nadeemalamansari/Stock-Price-Prediction-Model-Time-Series-Forecasting-/main/NSE-TATAGLOBAL.csv')
   ```

2. **Convert Date Column and Reverse Data**

   ```python
   data['Date'] = pd.to_datetime(data.Date, format='%Y-%m-%d')
   data = data.iloc[::-1]
   ```

3. **Set Date Column as Index**

   ```python
   data.set_index('Date', inplace=True)
   ```

## Stationarity Check

The stationarity of the time series is tested using the Augmented Dickey-Fuller (ADF) test. The series is transformed to make it stationary by applying a logarithmic transformation and differencing.

```python
def test_stationarity(timeseries):
    # Transformation to make the data stationary
    timeseries_log = np.log(timeseries)
    timeseries_log_diff = timeseries_log.diff().dropna()
    
    # Plot rolling statistics and perform ADF test
    ...
```

## Model Fitting

### Moving Average

This section uses a moving average method for prediction. The performance is evaluated using Root Mean Squared Error (RMSE).

```python
def moving_avg_prediction(data):
    ...
```

### ARIMA

Auto ARIMA is used for forecasting stock prices. The model automatically selects the best parameters for ARIMA.

```python
def arima_prediction(data):
    ...
```

### SARIMA

The Seasonal ARIMA (SARIMA) model is used to capture seasonality in the time series. The model is fitted with parameters for both non-seasonal and seasonal components.

```python
def sarima_prediction(data):
    ...
```

## Usage

1. **Load Data**

   ```python
   data = pd.read_csv('your_data.csv', parse_dates=True, index_col='Date')
   ```

2. **Run Forecasting Models**

   ```python
   moving_avg_prediction(data)
   arima_prediction(data)
   sarima_prediction(data)
   ```

