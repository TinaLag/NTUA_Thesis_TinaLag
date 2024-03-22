import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from os.path import join as pathjoin
from main import PROCESSED_DATA_FILENAME, RESAMPLED_DATA_FILENAME, read_csv_data, make_filename
import matplotlib.pyplot as plt
import xgboost as xgb
from configuration import prefix, suffix, LABEL_NAME, MEASURED_NAME, STEP, LAGS, OUT_STEPS, FORECASTER_NAME, BATCH_SIZE


def apply_scaling(scaler, data_):
    columns = data_.columns
    matrix = scaler.transform(data_)
    data_ = pd.DataFrame(matrix, columns=columns)
    return data_


def get_train_val_test(data_):
    n = len(data_)
    train_df_ = data_[0:int(n * 0.7)]
    val_df_ = data_[int(n * 0.7):int(n * 0.9)]
    test_df_ = data_[int(n * 0.9):]
    num_features_ = data_.shape[1]

    minmax_sc_ = StandardScaler()
    minmax_sc_.fit(train_df_)
    train_df_ = apply_scaling(minmax_sc_, train_df_)
    val_df_ = apply_scaling(minmax_sc_, val_df_)
    test_df_ = apply_scaling(minmax_sc_, test_df_)

    print(f"Len train_df {len(train_df_)} , val_df {len(val_df_)}, test_df {len(test_df_)}")
    print(f"Number of features: {num_features_} (as {train_df_.columns})")

    return train_df_, val_df_, test_df_, minmax_sc_


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df, label_columns=None, forecasters_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        self.forecasters_columns = forecasters_columns
        print(f"Label columns: {label_columns}")
        print(f"Forecasters columns: {forecasters_columns}")

        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        self.non_label_indices = 0  # [i for (i, name) in enumerate(train_df.columns) if name not in self.label_columns]

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.forecasters_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Forecasters column name(s): {self.forecasters_columns}',
            f'Label column name(s): {self.label_columns}'])


def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    forecasters = features[:, self.forecasters_slice, :]

    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)
        forecasters = tf.stack(
            [forecasters[:, :, self.column_indices[name]] for name in self.forecasters_columns],
            axis=-1)
        inputs = tf.stack(
            [inputs[:, :, self.column_indices[name]] for name in self.column_indices.keys() if
             name not in self.label_columns and name not in self.forecasters_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    forecasters.set_shape([None, self.label_width, None])

    return inputs, labels, forecasters

def split_feats_and_label(self, feats):
    inputs, labels, _ = self.split_window(feats)
    return inputs, labels

def split_forecasters(self, feats):
    _, _, forecasters = self.split_window(feats)
    return forecasters, forecasters

WindowGenerator.split_window = split_window
WindowGenerator.split_feats_and_label = split_feats_and_label
WindowGenerator.split_forecasters = split_forecasters


def plot(self, model=None, plot_col=None, max_subplots=3, filename=None):
    inputs, labels = self.example
    _, forecasts = self.example_forecasts

    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n + 1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, 0)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        plt.scatter(self.label_indices, forecasts[n, :, label_col_index],
                    edgecolors='k', label='Forecasts', c='#d966c6', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Time [h]')
    # plt.show()

    path = pathjoin("plots", "models_plots", prefix + suffix)
    savefile = make_filename([path, prefix + filename])

    plt.savefig(savefile)
    plt.close()


WindowGenerator.plot = plot


def plot_traditional(self, model=None, plot_col=None, max_subplots=3, filename=None, use_dmatrix=False):
    inputs, labels = self.example
    _, forecasts = self.example_forecasts

    feats = np.array(inputs)
    feats = np.reshape(feats, (feats.shape[0], feats.shape[1]))
    lab = np.array(labels)
    lab = np.reshape(lab, (lab.shape[0]))

    if use_dmatrix:
        dfeat = xgb.DMatrix(feats, label=lab)

    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n + 1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, 0)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        plt.scatter(self.label_indices, forecasts[n, :, label_col_index],
                    edgecolors='k', label='Forecasts', c='#d966c6', s=64)

        if model is not None:
            if use_dmatrix:
                predictions = model.predict(dfeat)
            else:
                predictions = model.predict(feats)
            plt.scatter(self.label_indices, predictions[n],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Time [h]')
    # plt.show()

    path = pathjoin("plots", "models_plots", prefix + suffix)
    savefile = pathjoin(path, prefix + filename)

    plt.savefig(savefile)
    plt.close()


WindowGenerator.plot_traditional = plot_traditional


def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=False,
        batch_size=BATCH_SIZE, )
    ds = ds.map(self.split_feats_and_label)

    return ds


WindowGenerator.make_dataset = make_dataset

def make_forecasts(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=False,
        batch_size=BATCH_SIZE, )
    dsf = ds.map(self.split_forecasters)

    return dsf


WindowGenerator.make_forecasts = make_forecasts


@property
def train(self):
    return self.make_dataset(self.train_df)


@property
def val(self):
    return self.make_dataset(self.val_df)


@property
def test(self):
    return self.make_dataset(self.test_df)

@property
def train_forecasts(self):
    return self.make_forecasts(self.train_df)


@property
def val_forecasts(self):
    return self.make_forecasts(self.val_df)


@property
def test_forecasts(self):
    return self.make_forecasts(self.test_df)


@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.test` dataset
        result = next(iter(self.test))
        # And cache it for next time
        self._example = result
    return result

@property
def example_forecasts(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example_forecasts', None)
    if result is None:
        # No example batch was found, so get one from the `.test` dataset
        result = next(iter(self.test_forecasts))
        # And cache it for next time
        self._example_forecasts = result
    return result



WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.train_forecasts = train_forecasts
WindowGenerator.val_forecasts = val_forecasts
WindowGenerator.test_forecasts = test_forecasts
WindowGenerator.example = example
WindowGenerator.example_forecasts = example_forecasts

target_name = LABEL_NAME
data = read_csv_data(DATA_FILENAME)
column_indices = {name: i for i, name in enumerate(data.columns)}
data = data[[MEASURED_NAME, LABEL_NAME, FORECASTER_NAME]]
if "diff" in prefix:
    data = data.diff()
    data = data.iloc[1:]

num_features = 1  # SCADA
train_df, val_df, test_df, minmax_sc = get_train_val_test(data)


multi_window = WindowGenerator(input_width=LAGS,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               label_columns=[target_name],
                               forecasters_columns=[FORECASTER_NAME],
                               train_df=train_df,
                               val_df=val_df,
                               test_df=test_df)

# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:multi_window.total_window_size]),
                           np.array(train_df[100:100 + multi_window.total_window_size]),
                           np.array(train_df[200:200 + multi_window.total_window_size])])

for example_inputs, example_labels in multi_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')

multi_window.plot(plot_col=MEASURED_NAME, filename="example.png")

print(multi_window)

CONV_WIDTH = 3
LABEL_WIDTH = 1
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=[target_name],
    forecasters_columns=[FORECASTER_NAME],
    train_df=train_df,
    val_df=val_df,
    test_df=test_df)
print('Wide_conv_window')
print(wide_conv_window)

wide_window = WindowGenerator(
    input_width=20, label_width=20, shift=1,
    label_columns=[target_name],
    forecasters_columns=[FORECASTER_NAME],
    train_df=train_df,
    val_df=val_df,
    test_df=test_df)

print('Wide_window')
print(wide_window)

for example_inputs, example_labels in wide_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')

wide_window.plot(plot_col=MEASURED_NAME, filename="example.png")

eval_metrics = ["loss", "MAE", "MAPE", "RMSE"]