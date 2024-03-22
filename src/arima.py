import datetime
# from main import f_main
from metrics import eval_metrics
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join as pathjoin
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from main import PROCESSED_DATA_FILENAME, RESAMPLED_DATA_FILENAME, read_csv_data
from configuration import LABEL_NAME, MEASURED_NAME


data = read_csv_data(PROCESSED_DATA_FILENAME)
column_indices = {name: i for i, name in enumerate(data.columns)}
# data = data[[MEASURED_NAME, LABEL_NAME]]
data = data.diff()
data = data. iloc[1:]

LATEST_NAME = "latest_15m"
LABEL_NAME = "cert_15m"
MEASURED_NAME = "SCADA_15m"

suffix = data.columns[0].split("_")[-1]

def save_csv_data(data, filename):
    data.to_csv(filename)

# num_features = 1 #SCADA
# train_df, val_df, test_df, minmax_sc = get_train_val_test(data)

# ARIMA models work best with stationary time series data. You may need to take difference(s) or use other methods to make your data stationary.
# Fit the ARIMA Model: Use the statsmodels library in Python to fit an ARIMA model. You'll need to specify the order of the ARIMA model, which consists
# of three components: p (order of autoregression), d (order of differencing), and q (order of moving average).

TEST_SIZE = 1000

def get_best_arima_model():
    # df_i, df_na = f_main()
    df = data.copy()

    #df.resample('H', convention = 'end').ffill()

    # df = df[~df.index.duplicated(keep='first')]
    # df = df.asfreq('H')

    #def arima_model(df):
        # p = 1  # Autoregressive order, number of lagged (or past) observations to consider for autoregression
        # d = 1  # Differencing order,  number of times the raw observations are differenced
        # q = 1  # Moving average order, size of the moving average window

    time_series = df[MEASURED_NAME]
    model = auto_arima(time_series, seasonal=False, trace=True, stepwise=True)      # max_p=3, max_q=3
    print(model.summary())

    X_train = df[MEASURED_NAME].iloc[:-TEST_SIZE]
    X_test = df[MEASURED_NAME].iloc[-TEST_SIZE:]

    best_model = ARIMA(X_train, order=model.order)

    best_model_fit = best_model.fit()

    return df, best_model_fit, X_train, X_test

def plots(df, best_model_fit, X_train, X_test):

    path = pathjoin("plots/arima_res", "diff_" + suffix)

    # Forecast future values
    forecast_steps = 4*24 #4*24 gia 15m
    forecast = best_model_fit.forecast(steps=forecast_steps)

    idx = X_test.iloc[:forecast_steps].index
    arma_forecast = pd.DataFrame(forecast)
    arma_forecast.index = X_test.iloc[:forecast_steps].index
    arma_forecast.set_index(idx, drop=True, inplace=True)

    plt.close()
    plt.plot(grid=True, figsize=(20, 5))
    plt.plot(X_test.iloc[:forecast_steps])
    plt.plot(arma_forecast)
    # plt.show()
    savefile = pathjoin(path, "forecast_" + suffix + ".png")
    plt.savefig(savefile)
    plt.close()

    #calculate and compare mae for arima forecast and latest (meteologica forecast)
    y_true = X_test.iloc[:forecast_steps]
    results_dict = eval_metrics(arma_forecast, y_true)
    results_df = pd.DataFrame(results_dict, index=['ARIMA'])

    y_latest = df.iloc[-TEST_SIZE:(-TEST_SIZE+forecast_steps)][LATEST_NAME]
    results_dict = eval_metrics(y_latest, y_true)
    results_df.loc['latest'] = results_dict
    current_datetime = datetime.datetime.utcnow()
    # results_df.to_csv('arima_' + str(current_datetime) + '.csv')
    arima_csv_file = pathjoin(path, "arima_" + str(current_datetime) + suffix + ".csv")
    # for diff: (path, "arima_" + str(current_datetime) + suffix + ".csv")
    save_csv_data(results_df, arima_csv_file)
    print(results_df)

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[MEASURED_NAME], label='Original Data ' + suffix)
    plt.plot(X_test[:forecast_steps].index, forecast, label='Forecast ' + suffix)
    plt.plot(df.index, df[LATEST_NAME], label='Latest ' + suffix)
    plt.plot(df.index, df[LABEL_NAME], label='cert ' + suffix)

    plt.xlim([df.index[-(TEST_SIZE+30)], df.index[-(TEST_SIZE-30)]])
    plt.xlabel('Time')
    plt.ylabel('Value')
    #plt.show()
    savefile = pathjoin(path, "arima_eval_" + suffix + ".png")
    plt.savefig(savefile)
    plt.close()


df, best_model_fit, X_train, X_test = get_best_arima_model()
plots(df, best_model_fit, X_train, X_test)

#return None

#df

    # # Generate synthetic time series data
    # np.random.seed(42)
    # time = np.arange(100)
    # data = 10 + np.sin(0.2 * time) + np.random.normal(scale=2, size=len(time))
    # # Create a DataFrame
    # df = pd.DataFrame({'Time': time, 'Data': data})
    # # Fit ARIMA model using auto_arima

    # # Print the summary of the model
    # model.summary()
    # # Fit the best model obtained
    # best_model = ARIMA(df['Data'], order=model.order)
    # best_model_fit = best_model.fit()
    # # Forecast future values
    # forecast_stepas = 20
    # forecast = best_model_fit.forecast(steps=forecast_steps)
    # # Plot the original data and forecasted values
    # plt.figure(figsize=(10, 6))
    # plt.plot(df['Time'], df['Data'], label='Original Data')
    # plt.plot(np.arange(len(df), len(df) + forecast_steps), forecast, label='Forecast')
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.show()