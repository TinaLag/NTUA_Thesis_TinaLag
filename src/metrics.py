from math import sqrt

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def eval_metrics(y_pred, y_true):
    mae = mean_absolute_error(y_pred,y_true)
    mape = mean_absolute_percentage_error(y_pred, y_true)
    mse = mean_squared_error(y_pred, y_true)
    rmse = sqrt(mse)

    results = dict(mae = mae, mape = mape, mse = mse, rmse = rmse)

    return results


