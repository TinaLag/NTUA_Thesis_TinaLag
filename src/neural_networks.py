import tensorflow as tf
import matplotlib.pyplot as plt
from os.path import join as pathjoin
from configuration import prefix, suffix, MAX_EPOCHS
import pandas as pd
import numpy as np
from window_generator import multi_window, wide_window, MEASURED_NAME, OUT_STEPS, \
    CONV_WIDTH, num_features, wide_conv_window
from main import save_csv_data, make_filename


def compile_and_fit(model, optimizer, window, patience=5):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='auto')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.MeanAbsoluteError(),
                            tf.keras.metrics.MeanAbsolutePercentageError(),
                            tf.keras.metrics.RootMeanSquaredError()],
                  # 'mean_absolute_error', 'mean_absolute_percentage_error', 'root_mean_squared_error'])
                  #   tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError(),  tf.keras.metrics.RootMeanSquaredError()
                  )


    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history



def plot_history(history, filename):
    # Get training history
    loss = history.history['loss']
    mae = history.history['mean_absolute_error']
    # val_set:
    val_loss = history.history['val_loss']
    val_mae = history.history['val_mean_absolute_error']

    epochs = range(1, len(loss) + 1)

    # Plot loss
    plt.plot(epochs, loss, label='Training loss')
    # Plot val loss
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()

    path = make_filename(["plots", "models_plots", prefix + suffix])
    savefile = pathjoin(path, prefix + filename + "_loss_" + suffix + ".png")

    plt.savefig(savefile)
    plt.close()

    # Plot mae
    plt.plot(epochs, mae, label='Training MAE')
    # Plot val mae
    plt.plot(epochs, val_mae, label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()

    path = make_filename(["plots", "models_plots", prefix + suffix])
    savefile = pathjoin(path, prefix + filename + "_mae_" + suffix + ".png")

    plt.savefig(savefile)
    plt.close()



def run():
    class MultiStepLastBaseline(tf.keras.Model):
        def call(self, inputs):
            return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

    # Baseline
    last_baseline = MultiStepLastBaseline()
    last_baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError(),
                                   tf.keras.metrics.RootMeanSquaredError()],
                          )

    multi_val_performance = {}
    test_set_performance = {}

    multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)  # validation
    test_set_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)

    multi_window.plot(last_baseline, plot_col=MEASURED_NAME, filename="baseline_last_" + suffix + ".png")

    df_multi_val_performance = pd.DataFrame.from_dict(multi_val_performance)
    df_test_set_performance = pd.DataFrame.from_dict(test_set_performance)

    path = make_filename(["csv_exports", prefix + suffix, "baseline_res", prefix + suffix])
    multi_val_performance_csv_file = pathjoin(path, prefix + "multi_val_performance_" + suffix + ".csv")
    test_set_performance_csv_file = pathjoin(path, prefix + "test_set_performance_" + suffix + ".csv")

    save_csv_data(df_multi_val_performance, multi_val_performance_csv_file)
    save_csv_data(df_test_set_performance, test_set_performance_csv_file)

    # Linear
    multi_linear_model = tf.keras.Sequential([
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * num_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])


    history = compile_and_fit(multi_linear_model, tf.keras.optimizers.Adam(learning_rate=0.01), multi_window)
    path = pathjoin("plots", "models_plots", prefix + suffix)
    savefile = make_filename([path, prefix + "linear_schema" + ".png"])
    tf.keras.utils.plot_model(multi_linear_model, to_file=savefile, show_shapes=True)
    hist_df = pd.DataFrame(history.history)

    path = make_filename(["csv_exports", prefix + suffix, "linear_res", prefix + suffix])
    hist_csv_file = pathjoin(path, prefix + "linear_history_" + suffix + ".csv")

    save_csv_data(hist_df, hist_csv_file)

    plot_history(history, "linear")

    multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
    test_set_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(multi_linear_model, plot_col=MEASURED_NAME, filename="linear_" + suffix + ".png")

    # multi_dense_model -> Fully connected
    multi_dense_model = tf.keras.Sequential([
        # Take the last time step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, dense_units]
        tf.keras.layers.Dense(512, activation='relu'),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * num_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    history = compile_and_fit(multi_dense_model, tf.keras.optimizers.Adam(), multi_window)
    hist_df = pd.DataFrame(history.history)
    path = pathjoin("plots", "models_plots", prefix + suffix)
    savefile = make_filename([path, prefix + "multi_dense" + ".png"])
    tf.keras.utils.plot_model(multi_dense_model, to_file=savefile, show_shapes=True)

    path = make_filename(["csv_exports", prefix + suffix, "multi_dense_res", prefix + suffix])
    hist_csv_file = pathjoin(path, prefix + "multi_dense_history_" + suffix + ".csv")

    save_csv_data(hist_df, hist_csv_file)

    plot_history(history, "multi_dense")

    multi_val_performance['FC'] = multi_dense_model.evaluate(multi_window.val)
    test_set_performance['FC'] = multi_dense_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(multi_dense_model, plot_col=MEASURED_NAME, filename="multi_dense_" + suffix + ".png")

    print("history loss fully connected: ")
    print(history.history['loss'])

    # CNN
    multi_conv_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * num_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    history = compile_and_fit(multi_conv_model, tf.keras.optimizers.Adam(), multi_window)
    hist_df = pd.DataFrame(history.history)

    path = pathjoin("plots", "models_plots", prefix + suffix)
    savefile = make_filename([path, prefix + "cnn" + ".png"])
    tf.keras.utils.plot_model(multi_conv_model, to_file=savefile, show_shapes=True)

    path = make_filename(["csv_exports", prefix + suffix, "cnn_res", prefix + suffix])
    hist_csv_file = pathjoin(path, prefix + "cnn_history_" + suffix + ".csv")

    save_csv_data(hist_df, hist_csv_file)

    plot_history(history, "cnn")

    multi_val_performance['CNN'] = multi_conv_model.evaluate(multi_window.val)
    test_set_performance['CNN'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
    wide_conv_window.plot(multi_conv_model, plot_col=MEASURED_NAME, filename="cnn_" + suffix + ".png")

    print("history loss cnn: ")
    print(history.history['loss'])

    # LSTM
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(4, return_sequences=True), # activation='ReLU'),
        tf.keras.layers.Dropout(0.2),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    history = compile_and_fit(lstm_model, tf.keras.optimizers.Adam(learning_rate=0.001), multi_window)
    hist_df = pd.DataFrame(history.history)

    path = pathjoin("plots", "models_plots", prefix + suffix)
    savefile = make_filename([path, prefix + "lstm" + ".png"])
    tf.keras.utils.plot_model(lstm_model, to_file=savefile, show_shapes=True)

    path = make_filename(["csv_exports", prefix + suffix, "lstm_res", prefix + suffix])
    hist_csv_file = pathjoin(path, prefix + "lstm_history_" + suffix + ".csv")

    save_csv_data(hist_df, hist_csv_file)

    plot_history(history, "lstm")

    multi_val_performance['LSTM'] = lstm_model.evaluate(multi_window.val)
    test_set_performance['LSTM'] = lstm_model.evaluate(multi_window.test, verbose=0)
    wide_window.plot(lstm_model, plot_col=MEASURED_NAME, filename="LSTM_" + suffix + ".png")

    print("history loss lstm: ")
    print(history.history['loss'])

    # GRU
    gru_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.GRU(4, return_sequences=True), #, activation='ReLU'),
        tf.keras.layers.Dropout(0.2),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    history = compile_and_fit(gru_model, tf.keras.optimizers.Adam(learning_rate=0.0005), multi_window)
    hist_df = pd.DataFrame(history.history)

    path = pathjoin("plots", "models_plots", prefix + suffix)
    savefile = make_filename([path, prefix + "gru" + ".png"])
    tf.keras.utils.plot_model(gru_model, to_file=savefile, show_shapes=True)

    path = make_filename(["csv_exports", prefix + suffix, "gru_res", prefix + suffix])
    hist_csv_file = pathjoin(path, prefix + "gru_history_" + suffix + ".csv")

    save_csv_data(hist_df, hist_csv_file)

    plot_history(history, "gru")

    multi_val_performance['GRU'] = gru_model.evaluate(multi_window.val)
    test_set_performance['GRU'] = gru_model.evaluate(multi_window.test, verbose=0)
    wide_window.plot(gru_model, plot_col=MEASURED_NAME, filename="GRU_" + suffix + ".png")

    print("history loss gru: ")
    print(history.history['loss'])

    # save and convert to dataframes
    df_multi_val_performance = pd.DataFrame.from_dict(multi_val_performance)
    df_test_set_performance = pd.DataFrame.from_dict(test_set_performance)

    save_csv_data(df_multi_val_performance, pathjoin("csv_exports", prefix + suffix, prefix + "multi_val_performance" + suffix + ".csv"))
    save_csv_data(df_test_set_performance, pathjoin("csv_exports", prefix + suffix, prefix + "test_set_performance" + suffix + ".csv"))

    # plot metrics for models
    x = np.arange(len(test_set_performance))
    width = 0.3

    metric_name = 'mean_absolute_error'
    metric_index = lstm_model.metrics_names.index(metric_name)
    val_mae = [v[metric_index] for v in multi_val_performance.values()]
    test_mae = [v[metric_index] for v in test_set_performance.values()]

    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=test_set_performance.keys(),
               rotation=45)
    plt.ylabel(f'MAE (average over all times and outputs)')
    _ = plt.legend()
    # plt.show()

    path = make_filename(["plots", "metrics_plots", prefix + suffix])
    savefile = pathjoin(path, prefix + metric_name + "_" + suffix + ".png")

    plt.savefig(savefile)
    plt.close()

    metric_name = 'mean_absolute_percentage_error'
    metric_index = lstm_model.metrics_names.index(metric_name)
    val_mae = [v[metric_index] for v in multi_val_performance.values()]
    test_mae = [v[metric_index] for v in test_set_performance.values()]

    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=test_set_performance.keys(),
               rotation=45)
    plt.ylabel(f'MAPE (average over all times and outputs)')
    _ = plt.legend()
    # plt.show()

    path = make_filename(["plots", "metrics_plots", prefix + suffix])
    savefile = pathjoin(path, prefix + metric_name + "_" + suffix + ".png")

    plt.savefig(savefile)
    plt.close()

    metric_name = 'root_mean_squared_error'
    metric_index = lstm_model.metrics_names.index(metric_name)
    val_mae = [v[metric_index] for v in multi_val_performance.values()]
    test_mae = [v[metric_index] for v in test_set_performance.values()]

    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=test_set_performance.keys(),
               rotation=45)
    plt.ylabel(f'RMSE (average over all times and outputs)')
    _ = plt.legend()
    # plt.show()

    path = make_filename(["plots", "metrics_plots", prefix + suffix])
    savefile = pathjoin(path, prefix + metric_name + "_" + suffix + ".png")

    plt.savefig(savefile)
    plt.close()


if __name__ == "__main__":
    run()