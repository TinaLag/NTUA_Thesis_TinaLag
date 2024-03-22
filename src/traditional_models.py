import numpy as np
import xgboost as xgb
import lightgbm as lgb
from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import tensorflow as tf
from configuration import suffix, prefix
from window_generator import multi_window, eval_metrics, MEASURED_NAME
from os.path import join as pathjoin
from main import save_csv_data, make_filename


def flatten(input):
    return [item for row in input for item in row]

def get_dataset(target_set, target_forecast_set):
    feats = []
    lab = []
    for inputs, labels in target_set:
        feats.append(inputs)
        lab.append(labels)

    #[print(len(x)) for x in feats]
    feats = flatten(feats)
    lab = flatten(lab)
    feats = np.array(feats)
    lab = np.array(lab)

    feats = np.reshape(feats, (feats.shape[0], feats.shape[1]))
    lab = np.reshape(lab, (lab.shape[0]))

    forecasts = []
    for _, frc in target_forecast_set:
        forecasts.append(frc)

    forecasts = flatten(forecasts)
    forecasts = np.array(forecasts)
    forecasts = np.reshape(forecasts, (forecasts.shape[0]))

    print(feats.shape)
    print(lab.shape)
    print(forecasts.shape)
    return feats, lab, forecasts


def evaluate_traditional_model(multi_val_performance, test_set_performance,
                               model_, model_name, model_key,
                               train_feats, val_feats, test_feats,
                               train_lab, val_lab, test_lab, use_dmatrix=False):
    mape = tf.keras.metrics.mean_absolute_percentage_error
    mae = tf.keras.metrics.mean_absolute_error
    mse = tf.keras.metrics.mean_squared_error

    train_pred = model_.predict(train_feats) if model_ is not None else train_feats
    val_pred = model_.predict(val_feats) if model_ is not None else val_feats
    test_pred = model_.predict(test_feats) if model_ is not None else test_feats


    train_perf = {0: mse(train_lab, train_pred).numpy(),
                  1: mae(train_lab, train_pred).numpy(),
                  2: mape(train_lab, train_pred).numpy(),
                  3: np.sqrt(mse(train_lab, train_pred))}

    val_perf = {0: mse(val_lab, val_pred).numpy(),
                  1: mae(val_lab, val_pred).numpy(),
                  2: mape(val_lab, val_pred).numpy(),
                  3: np.sqrt(mse(val_lab, val_pred))}


    test_perf = {0: mse(test_lab, test_pred).numpy(),
                  1: mae(test_lab, test_pred).numpy(),
                  2: mape(test_lab, test_pred).numpy(),
                  3: np.sqrt(mse(test_lab, test_pred))}

    print(model_name)
    print("Train performance")
    [print(eval_metrics[x], '=', train_perf[x]) for x in train_perf]
    print("Val performance")
    [print(eval_metrics[x], '=', val_perf[x]) for x in val_perf]
    print("Test performance")
    [print(eval_metrics[x], '=', test_perf[x]) for x in test_perf]

    multi_val_performance[model_key] = val_perf
    test_set_performance[model_key] = test_perf


    multi_window.plot_traditional(model_, plot_col=MEASURED_NAME, filename= model_key + "_" + suffix + ".png", use_dmatrix=use_dmatrix)

    return multi_val_performance, test_set_performance

def run():

    multi_val_performance = read_csv(
        pathjoin("csv_exports", prefix + suffix, prefix + "multi_val_performance" + suffix + ".csv"),
        index_col=False).drop(
        columns=['Unnamed: 0'])
    test_set_performance = read_csv(
        pathjoin("csv_exports", prefix + suffix, prefix + "test_set_performance" + suffix + ".csv"),
        index_col=False).drop(
        columns=['Unnamed: 0'])

    train_feats, train_lab, train_forecasts = get_dataset(multi_window.train, multi_window.train_forecasts)
    val_feats, val_lab, val_forecasts = get_dataset(multi_window.val, multi_window.val_forecasts)
    test_feats, test_lab, test_forecasts = get_dataset(multi_window.test, multi_window.test_forecasts)



    # random forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(train_feats, train_lab)
    multi_val_performance, test_set_performance = evaluate_traditional_model(multi_val_performance, test_set_performance, rf_model, "Random Forest", "RF",
                                                                             train_feats, val_feats, test_feats, train_lab,
                                                                             val_lab, test_lab, use_dmatrix=False)


    # XGBoost
    dtrain = xgb.DMatrix(train_feats, label=train_lab)
    dval = xgb.DMatrix(val_feats, label=val_lab)
    dtest = xgb.DMatrix(test_feats, label=test_lab)

    params = {
        'max_depth': 3,
        'seed': 42,
    }

    num_rounds = 100
    xgb_model = xgb.train(params, dtrain, num_rounds)
    multi_val_performance, test_set_performance = evaluate_traditional_model(multi_val_performance, test_set_performance, xgb_model, "XGB", "XGB",
                                                                             dtrain, dval, dtest, train_lab, val_lab,
                                                                             test_lab, use_dmatrix=True)


    # LGBMRegressor
    lgb_model = lgb.LGBMRegressor(metric='rmse')
    lgb_model.fit(train_feats, train_lab)
    multi_val_performance, test_set_performance = evaluate_traditional_model(multi_val_performance, test_set_performance, lgb_model, "LightLGB", "LGB",
                                                                             train_feats, val_feats, test_feats, train_lab,
                                                                             val_lab, test_lab, use_dmatrix=False)


    multi_val_performance, test_set_performance = evaluate_traditional_model(multi_val_performance, test_set_performance,None, "Forecasters", "Frc",
                                                                             train_forecasts, val_forecasts, test_forecasts, train_lab,
                                                                             val_lab, test_lab, use_dmatrix=False)

    # add traditional models' results
    multi_val_performance.index = eval_metrics
    multi_val_performance = multi_val_performance



    csv_file = make_filename(["plots", "models_plots", prefix + suffix, "final_all_res", prefix + "multi_val_performance_" + suffix + ".csv"])
    save_csv_data(multi_val_performance, csv_file)
    multi_val_performance = multi_val_performance.to_dict()

    print("multi val performance:")
    print(multi_val_performance)

    test_set_performance.index = eval_metrics
    test_set_performance = test_set_performance

    csv_file = make_filename(["plots", "models_plots", prefix + suffix, "final_all_res", prefix + "test_set_performance_" + suffix + ".csv"])
    save_csv_data(test_set_performance, csv_file)
    test_set_performance = test_set_performance.to_dict()


    # metrics
    x = np.arange(len(test_set_performance))
    width = 0.3

    # MAE
    val_mae = []
    val_model_names = []
    for model_name, metrics in multi_val_performance.items():
        val_model_names.append(model_name)
        val_mae.append(metrics['MAE'])

    val_mae = [v['MAE'] for v in multi_val_performance.values()]

    test_mae = []
    test_model_names = []
    for model_name, metrics in test_set_performance.items():
        test_model_names.append(model_name)
        test_mae.append(metrics['MAE'])

    test_mae = [v['MAE'] for v in test_set_performance.values()]

    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=test_set_performance.keys(),
               rotation=45)
    plt.ylabel(f'MAE (average over all times and outputs)')
    _ = plt.legend()
    # plt.show()

    savefile = make_filename(["plots", "models_plots", prefix + suffix, "final_all_res", prefix + "mae_" + suffix + "_final" + ".png"])
    plt.savefig(savefile)
    plt.close()

    # MAPE
    val_mae = []
    val_model_names = []
    for model_name, metrics in multi_val_performance.items():
        val_model_names.append(model_name)
        val_mae.append(metrics['MAPE'])

    val_mae = [v['MAPE'] for v in multi_val_performance.values()]

    test_mae = []
    test_model_names = []
    for model_name, metrics in test_set_performance.items():
        test_model_names.append(model_name)
        test_mae.append(metrics['MAPE'])

    test_mae = [v['MAPE'] for v in test_set_performance.values()]

    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=test_set_performance.keys(), rotation=45)
    plt.ylabel(f'MAPE (average over all times and outputs)')
    _ = plt.legend()
    # plt.show()
    savefile = make_filename(["plots",  "models_plots", prefix + suffix, "final_all_res", prefix + "mape_" + suffix + "_final" + ".png"])
    plt.savefig(savefile)
    plt.close()


    # RMSE
    val_mae = []
    val_model_names = []
    for model_name, metrics in multi_val_performance.items():
        val_model_names.append(model_name)
        val_mae.append(metrics['RMSE'])

    val_mae = [v['RMSE'] for v in multi_val_performance.values()]

    test_mae = []
    test_model_names = []
    for model_name, metrics in test_set_performance.items():
        test_model_names.append(model_name)
        test_mae.append(metrics['RMSE'])

    test_mae = [v['RMSE'] for v in test_set_performance.values()]

    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=test_set_performance.keys(), rotation=45)
    plt.ylabel(f'RMSE (average over all times and outputs)')
    _ = plt.legend()
    # plt.show()
    savefile = make_filename(["plots",  "models_plots", prefix + suffix, "final_all_res", prefix + "rmse_" + suffix + "_final" + ".png"])
    plt.savefig(savefile)
    plt.close()

if __name__ == '__main__':
    run()

