import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import exists
import logging
from datetime import datetime
from os.path import join as pathjoin
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib


# plt.ion()
# import sklearn as sk
# % matplotlib inline
# import tkinter as tk

matplotlib.use('TkAgg')  # Use the Tkinter backend

log_name = pathjoin("logs", ("log_" + str(datetime.now()) + ".txt").replace(":","_").replace(".","_"))
logging.basicConfig(filename=log_name, level=logging.DEBUG)
logger = logging.getLogger("main")
stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.addHandler(stdout_handler)

RAW_DATA_FILENAME = "data/data_15m.csv"
PROCESSED_DATA_FILENAME = "data/data_15m_proc.csv"
RESAMPLED_DATA_FILENAME = "data/data_60m.csv"
FEATURE_DATA_FILENAME = "data/data_60m_feat.csv"

RUN_FROM_SCRATCH = False


def make_filename(filename_parts):
    target_path = pathjoin(*filename_parts)
    if '.' not in target_path:
        target_dir = target_path
    else:
        target_dir, filename = os.path.split(target_path)
    if target_dir != "" and not os.path.exists(target_dir):
        os.makedirs(target_dir)

    return target_path


def read_csv_data(filename):
    data = pd.read_csv(filename, index_col=False)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data.set_index('Timestamp', drop=True, inplace=True)
    logger.info("Finished read file")
    return data


def save_csv_data(data, filename):
    data.to_csv(filename)
    logger.info("Finished save file")

def remove_consec_zero_rows(data, target_column):
    # Zero values for night time need to be removed from data frame
    data['value_grp'] = (data[target_column].diff(1) != 0).astype('int').cumsum()  # Find all consecutive zeroes and create groups for every time another cont starts

    CONSEC_ZEROS = 5
    group_names = np.unique(data['value_grp'].values)  # Every unique group of zeros that was produced above
    groups_of_zeros = [x for x in group_names if (len(data[data['value_grp'] == x]) > CONSEC_ZEROS)]  # Keep all groups that have more than set number of consecutive zeros, i.e 5

    logger.info(f"Length before {len(data)}")
    id_not_zeros = [i for i, x in enumerate(data['value_grp']) if x not in groups_of_zeros]  # Find "place" of each group in the data meaning timestamps
    data = data.iloc[id_not_zeros, :]  # Remove those timestamps from dataframe
    logger.info(f"Length after {len(data)}")
    logger.info("Finished remove consec zeros")
    return data


def fix_error_data(data):
    return data


def find_outliers_IQR(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)

    IQR = q3 - q1
    logger.info(f"Q1 {q1} and q2 {q3}")

    threshold = 1.2
    outliers = data[((data < (q1 - threshold * IQR)) | (data > (q3 + threshold * IQR)))]

    # outlier = []
    # for x in data[LABEL_NAME]:
    #     if ((x> upper_bound) or (x<lower_bound)):
    #          outlier.append(x)
    # print(' outlier in the dataset is', outlier)

    return outliers


def preprocessing(data):
    if RUN_FROM_SCRATCH or not exists(PROCESSED_DATA_FILENAME):
        data.columns = ['cert_final_15m', 'cert_15m', 'SCADA_15m', 'avail_15m', 'latest_15m', 'irrad_15m']

        data['avail_15m'].replace(np.nan, 100, inplace=True)  # Replace availability nan with 100

        data.replace(np.nan, 0, inplace=True)  # Replace all Nan values with 0
        logger.info("Finished replace NAN")

        data = remove_consec_zero_rows(data, "cert_15m")

        logger.info(f"1\n {data.describe().T}")

        # Production is corrected using availability
        df = data.copy()
        MEASURED_NAME = 'SCADA_15m'
        LABEL_NAME = 'cert_15m'
        df.rename(columns={MEASURED_NAME: 'init_' + MEASURED_NAME}, inplace=True)
        df[MEASURED_NAME] = df['init_' + MEASURED_NAME] * df['avail_15m'] / 100
        df.rename(columns={LABEL_NAME: 'init_' + LABEL_NAME}, inplace=True)
        df[LABEL_NAME] = df['init_' + LABEL_NAME] * df['avail_15m'] / 100

        logger.info("Finished multiply avail")
        logger.info(f"2\n {df.describe().T}")

        outliers = find_outliers_IQR(df[MEASURED_NAME])
        indices = outliers.index

        logger.info(outliers)
        logger.info("number of outliers: " + str(len(outliers)))
        logger.info("max outlier value: " + str(outliers.max()))
        logger.info("min outlier value: " + str(outliers.min()))
        logger.info(f"length before outlier drop {len(df)}")

        df.loc[indices, MEASURED_NAME] = 0
        logger.info(f"length after outlier drop {len(df)}")

        # Zero values that remain correspond to actual production and need to be interpolated
        df.loc[df[LABEL_NAME] == 0, :] = np.nan  # Zeros need to be replaced with nan values for interpolation
        df.interpolate(method='linear', inplace=True)  # Intepolation using spline method
        logger.info("Finished interpolate")

        save_csv_data(df, PROCESSED_DATA_FILENAME)
    else:
        df = read_csv_data(PROCESSED_DATA_FILENAME)
        df.drop(columns='value_grp', inplace=True)
    return df


def resample(data):
    if RUN_FROM_SCRATCH or not exists(RESAMPLED_DATA_FILENAME):
        data_60m = data.resample('H').sum()
        new_name_dict = {x: x.replace('15m', '60m') for x in data_60m.columns}
        data_60m.rename(columns=new_name_dict, inplace=True)
        logger.info(data_60m.head())

        data_60m = remove_consec_zero_rows(data_60m, "cert_60m")

        save_csv_data(data_60m, RESAMPLED_DATA_FILENAME)
    else:
        data_60m = read_csv_data(RESAMPLED_DATA_FILENAME)
    return data_60m


def prepare_features(data):
    print("Not implemented ")

    if RUN_FROM_SCRATCH or not exists(FEATURE_DATA_FILENAME):
        # TODO; Make difference features

        save_csv_data(data, FEATURE_DATA_FILENAME)
    else:
        data = read_csv_data(FEATURE_DATA_FILENAME)
    return data


def get_plot_timeseries(data, savefile):
    plt.tight_layout()

    data.iloc[0:2000].plot(figsize=(15, 5))
    plt.title('Time series plot')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=3)
    # Adjusting layout to make space for legend below the graph
    plt.subplots_adjust(bottom=0.25)

    plt.savefig(savefile)
    plt.close()


def get_pairplot(data):
    plt.tight_layout()

    add_to_name = data.columns[0].split("_")[-1]
    savefile = make_filename(["plots", "pairplot_" + add_to_name + ".png"])
    # pairplot -> to fix the future warnings might need pip install: --force-reinstall https://github.com/mwaskom/seaborn/archive/refs/heads/master.tar.gz
    if RUN_FROM_SCRATCH or not exists(savefile):
        to_drop_cols = [x for x in data.columns if
                        any(ext in x for ext in ['cert_final', 'avail', 'irrad', 'value_grp', 'init'])]
        data.drop(columns=to_drop_cols, inplace=True)
        sns.pairplot(data)
        plt.savefig(savefile)
        plt.close()
        logger.info("Finished pairplot " + add_to_name)


def get_statistics(data):
    add_to_name = data.columns[0].split("_")[-1]
    savefile = make_filename(["csv_exports", "Statistics_" + add_to_name + ".csv"])
    if RUN_FROM_SCRATCH or not exists(savefile):
        stats = data.describe().T
        stats['median'] = data.median()
        stats['mode'] = data.mode().T[0]
        to_drop_cols = [x for x in stats.index if any(ext in x for ext in ['cert_final', 'avail', 'irrad'])]
        stats.drop(to_drop_cols, inplace=True)
        logger.info(f"Stats\n{stats}")

        stats['variance'] = data.var()
        stats['Standard_VariationD'] = data.std()
        logger.info(f"All_stats\n{stats}")

        save_csv_data(stats, savefile)
        logger.info("Finished statistics " + add_to_name)
        print(stats)


def get_boxplots(data):
    plt.tight_layout()

    suffix = data.columns[0].split("_")[-1]
    savefile = make_filename(["plots", "boxplot_" + suffix + ".png"])
    if RUN_FROM_SCRATCH or not exists(savefile):
        to_drop_cols = [x for x in data.columns if
                        any(ext in x for ext in ['cert_final', 'avail', 'irrad', 'value_grp', 'init'])]
        data.drop(columns=to_drop_cols, inplace=True)
        boxplot = data.boxplot(grid=True)
        boxplot.set_title('data boxplot')
        plt.savefig(savefile)
        plt.close()
        logger.info("Finished boxplot " + suffix)


def get_rolling_statistics(data):
    # a rolling mean or rolling average (also known as a moving average), which is simply the mean of a specific time series data column value
    # over a specified number of periods
    data = data.resample('h').sum()
    files = [x for x in os.listdir("plots") if "Rolling_average" in x]

    if RUN_FROM_SCRATCH or len(files) < 3:
        data = data.iloc[0:1000]
        data['simple_moving_average_6hours'] = data['cert_60m'].rolling(6).mean()
        data['simple_moving_average_24hours'] = data['cert_60m'].rolling(24).mean()
        data['simple_moving_average_1month'] = data['cert_60m'].rolling(24 * 30).mean()

        data['Timestamp'] = data.index.copy()

        # fig, ax = plt.subplots()
        # ax2 = ax.twinx()
        plt.tight_layout()

        # 6hours
        data.plot(x='Timestamp', y=['cert_60m', 'simple_moving_average_6hours'], kind='line', grid=True,
                  figsize=(20, 5))
        plt.title("Rolling average 6H")
        plt.savefig(make_filename(["plots", "Rolling_average_6H.png"]))
        plt.close()
        logger.info("Finished 6H rolling average")

        # 24hours
        data.plot.line(x='Timestamp', y=['cert_60m', 'simple_moving_average_24hours'], grid=True, figsize=(20, 5))
        plt.title("Rolling average 24H")
        plt.savefig(make_filename(["plots", "Rolling_average_24H.png"]))
        plt.close()
        logger.info("Finished 24H rolling average")

        # 1month
        data.plot.line(x='Timestamp', y=['cert_60m', 'simple_moving_average_1month'], grid=True, figsize=(20, 5))
        plt.title("Rolling average 1 month")
        plt.savefig(make_filename(["plots", "Rolling_average_1M.png"]))
        plt.close()
        logger.info("Finished 1M rolling average")


def get_auto_corr(data):
    # Autocorrelation Function (ACF): This function measures the correlation between a time series and its lagged values. It helps identify seasonality and lag effects.
    suffix = data.columns[0].split("_")[-1]
    savefile = make_filename(["plots", "auto_corr_" + suffix + ".png"])
    savefile2 = make_filename(["plots", "auto_cor_80_bins.png"])
    if RUN_FROM_SCRATCH or not exists(savefile):
        plt.tight_layout()
        pd.plotting.autocorrelation_plot(data['cert_' + suffix]).plot(grid=True, figsize=(20, 5))
        plt.title('Auto correlation plot')
        plt.savefig(savefile)
        plt.close()
        logger.info("Finished auto correlation plot " + suffix)

        # plt.figure(figsize=(20, 5))  # Adjust the figure size if needed
        # pd.plotting.autocorrelation_plot(data['cert_' + suffix], lags=80).plot(grid =True, figsize = (20,5))              # lags unknown keyword

        plt.figure(figsize=(20, 5))  # Adjust the figure size if needed
        plt.tight_layout()
        plt.acorr(data['cert_' + suffix], maxlags=80, usevlines=False, marker="o", linestyle="-")
        plt.title('Autocorrelation Plot with 80 Bins')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid(True)
        plt.savefig(savefile2)
        plt.close()
        logger.info("Finished auto correlation plot 80 bins")


def get_lags(data):
    suffix = data.columns[0].split("_")[-1]
    LAGS_FILENAME = make_filename(["csv_exports", "lags" + suffix])
    IMPORTANT_LAGS_FILENAME = make_filename(["csv_exports", "Important_lags" + suffix])
    if RUN_FROM_SCRATCH or not exists(LAGS_FILENAME):
        suffix = data.columns[0].split("_")[-1]
        # df_lags shows lags in importance order from auto correlation function
        lags = sm.tsa.acf(data["cert_" + suffix])
        df_lags = pd.DataFrame(lags).abs().sort_values(by=0, axis=0, ascending=False)
        df_lags.columns = ['auto_corr']
        df_lags.rename_axis('lags')

        dict_lags = {}
        for index, row in df_lags.iterrows():
            dict_lags[index] = row['auto_corr']

        logger.info("Finished finding lags")

        # keep only important lads, those with autocorrelation >= 0.75
        threshold = 0.75
        # important_lags = df_lags.loc[(df_lags['auto_corr'] < 1) & (df_lags['auto_corr'] >= threshold)]
        # or:
        mask = (df_lags < 1) & (df_lags >= threshold)
        important_lags = df_lags.loc[mask.all(axis=1)]

        dict_import_lags = {}
        for index, row in important_lags.iterrows():
            dict_import_lags[index] = row['auto_corr']

        logger.info("Finished finding important lags")

        if suffix == "60m":
            for key, value in dict_import_lags.items():
                data[f'cert_{suffix}_lag_{key}'] = data['cert_60m'].shift(key)
                pd.plotting.lag_plot(data[f'cert_{suffix}_lag_{key}']).plot(grid=True, figsize=(20, 5))
                plt.title(f'Lag plot for lag={key}')
                savefile = make_filename(["plots", f"lag_plot_{key}_Hour_" + suffix + ".png"])
                plt.savefig(savefile)  # f'lag_plot_{key}.png'
                plt.close()

            i_lags = [6, 24]
            for i in i_lags:
                data[f'cert_60m_lag_{i}'] = data['cert_60m'].shift(i)
                pd.plotting.lag_plot(data[f'cert_60m_lag_{i}']).plot(grid=True, figsize=(20, 5))
                plt.title(f'Lag plot for lag={i}')
                savefile = make_filename(["plots", f"lag_plot_{i}_Hour_" + suffix + ".png"])
                plt.savefig(savefile)
                plt.close()

        logger.info("Finished plotting important lags")

        save_csv_data(df_lags, LAGS_FILENAME)
        save_csv_data(important_lags, IMPORTANT_LAGS_FILENAME)

    return data


def get_seas_decomp(data):
    suffix = data.columns[0].split("_")[-1]
    savefile = make_filename(["plots", "seasonal_decomposition_" + suffix + ".png"])
    if RUN_FROM_SCRATCH or not exists(savefile):
        data_ = data.diff(1).copy()
        seas_decomp = seasonal_decompose(data['cert_' + suffix], period=24)  # den xreiazetai model by default additive
        # print(seas_decomp.trend)
        # print(seas_decomp.seasonal)
        # print(seas_decomp.resid)
        # print(seas_decomp.observed)
        seas_decomp = seasonal_decompose(data_.loc[data_.index[-3000]:, 'cert_' + suffix], period=24)
        plt.rcParams['figure.figsize'] = (20, 8)
        seas_decomp.plot()
        plt.title('Seasonal Decomposition')
        plt.savefig(savefile)
        plt.close()
        logger.info("Finished seasonal decomp plot " + suffix)


def get_histogram(data):
    suffix = data.columns[0].split("_")[-1]
    savefile = make_filename(["plots", "histogram_" + suffix + ".png"])
    if RUN_FROM_SCRATCH or not exists(savefile):
        # hist = data.hist()
        data['cert_' + suffix].hist(bins=10, color='green', alpha=0.7)
        plt.xlabel('cert_' + suffix)
        plt.ylabel('Frequency')
        plt.title('Histogram')
        plt.savefig(savefile)
        plt.close()


def get_2d_plot(data):
    suffix = data.columns[0].split("_")[-1]
    savefile = make_filename(["plots", "distrib_" + suffix + ".png"])
    if RUN_FROM_SCRATCH or not exists(savefile):
        plt.hist2d(data['cert_' + suffix], data['SCADA_' + suffix], bins=(50, 50))
        plt.colorbar()
        plt.xlabel('Certified data')
        plt.ylabel('SCADA data')
        ax = plt.gca()
        ax.axis('tight')
        plt.savefig(savefile)
        plt.close()


def statical_analysis_and_plots(data, run_for_60m=False):
    get_pairplot(data)
    get_statistics(data)
    get_boxplots(data)
    if run_for_60m:
        get_rolling_statistics(data)
    get_auto_corr(data)
    get_seas_decomp(data)
    get_histogram(data)
    get_2d_plot(data)


# new main
def get_raw_statistics():
    data = read_csv_data(RAW_DATA_FILENAME)
    # EDA before csv_exports
    # plot data
    get_plot_timeseries(data.drop(columns=[]), pathjoin("plots", "raw.png"))

    # dataset shape
    print(data)
    data.columns = ['cert_final_15m', 'cert_15m', 'SCADA_15m', 'avail_15m', 'latest_15m', 'irrad_15m']
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    # get statistics
    logger.info("Get RAW statistics ")
    get_statistics(data)
    logger.info("Finished RAW statistics ")

    df = data
    statical_analysis_and_plots(df, False)

    return

# new main
def f_main():
    data = read_csv_data(RAW_DATA_FILENAME)

    data = preprocessing(data)
    get_plot_timeseries(data.drop(columns=['avail_15m', 'cert_final_15m', 'init_cert_15m', 'init_SCADA_15m']),
                        pathjoin("plots", "preproc.png"))

    logger.info("Get PREPROC statistics ")
    get_statistics(data)
    logger.info("Finished PREPROC statistics ")

    data_60m = resample(data)
    data_60m = prepare_features(data_60m)

    dataframes = [data, data_60m]
    plot_settings = [False, True]

    for (df, runs_for_60m) in zip(dataframes, plot_settings):
        statical_analysis_and_plots(df, runs_for_60m)

    get_lags(data)
    get_lags(data_60m)

    # # using describe() function we can get q1, q3, middle. It can also provide statistics for dataframe
    # data.describe()
    # q1_for_cert = data.describe().loc['25%', 'cert_15m']
    # q2_for_cert = data.describe().loc['50%', 'cert_15m']
    # q3_for_cert = data.describe().loc['75%', 'cert_15m']

    # # pearson coefficient python gia oles tis steiles -> einai gia linear corr
    # correlation_matrix = np.corrcoef(data, rowvar=False)
    # correlation_df = pd.DataFrame(correlation_matrix, columns=data.columns, index=data.columns)
    # correlation_df
    # # np.corrcoef(data['cert_15m'],data['SCADA_15m'])

    return data, data_60m


if __name__ == "__main__":
    get_raw_statistics()
    data, data_60m = f_main()


#     #rolling statistics
#     #a rolling mean or rolling average (also known as a moving average), which is simply the mean of a specific time series data column value
#     #over a specified number of periods
#     data['simple_moving_average_6hours'] = data['cert_60m'].rolling(6).mean()
#     data['simple_moving_average_24hours'] = data['cert_60m'].rolling(24).mean()
#     data['simple_moving_average_1month'] = data['cert_60m'].rolling(24*30).mean()
#     # mono gia 15m xreiazetai to groupby (for aggregation in general): group6H = data.groupby('cert_60m').rolling(6).mean()
#
#     #plots or rolling average
#     data['Timestamp'] = data.index.copy()
#
#     #6hours
#     SMA6H = data.loc[:, 'simple_moving_average_6hours']
#     SMA6H.plot(grid = True, figsize = (20,5))
#     plt.title('Simple moving average for 6 hours')
#     plt.savefig('Simple_moving_average_6_hours.png')
#     plt.close()
#
#     #1month
#     SMA1M = data.loc[:, 'simple_moving_average_1month']
#     SMA1M.plot(grid = True, figsize = (20,5))
#     plt.title('Simple moving average for 1 month')
#     plt.savefig('Simple_moving_average_1_month.png')
#     plt.close()
#
#
#     # autocorrelation
#     # 3 differrent functions for autocorrelation: pd.plotting.autocorrelation_plot(), pd.Series.autocorr(), and plt.acorr()
#     # Autocorrelation Function (ACF): This function measures the correlation between a time series and its lagged values. It helps identify seasonality and lag effects.
#     # auto_cor = pd.plotting.autocorrelation_plot(data['cert_60m'])
#     # auto_cor.plot(grid =True, figsize = (20,5))
#     # plt.title('Auto correlation plot')
#     # plt.savefig('auto_cor.png')
#     # plt.close()
#
#     # Create an autocorrelation plot with a specific number of bins
#     # For example, let's set the number of bins to 20
#     # plt.figure(figsize=(20, 5))  # Adjust the figure size if needed
#     # pd.plotting.autocorrelation_plot(data['cert_60m'], lags=80)
#     # plt.title('Autocorrelation Plot with 80 Bins')
#     # plt.xlabel('Lag')
#     # plt.ylabel('Autocorrelation')
#     # plt.grid(True)
#     # plt.savefig('auto_cor.png')
#     # plt.close()
#
#     # plt.figure(figsize=(20, 5))  # Adjust the figure size if needed
#     # plt.acorr(data['cert_60m'], maxlags=80, usevlines=False, marker="o", linestyle="-")
#     # plt.title('Autocorrelation Plot with 80 Bins')
#     # plt.xlabel('Lag')
#     # plt.ylabel('Autocorrelation')
#     # plt.grid(True)
#     # plt.savefig('auto_cor.png')
#     # plt.close()
#     #
#     # #df_lags shows lags in importance order from auto correlation function
#     # lags = sm.tsa.acf(data['cert_60m'])
#     # df_lags = pd.DataFrame(lags).abs().sort_values(by=0, axis=0, ascending=False)
#     # df_lags.columns = ['auto_corr']
#     # df_lags.rename_axis('lags')
#     # #lags.loc[:,'auto_corr']
#     #
#     # #dict of lags
#     # dict_lags = {}
#     #
#     # for index, row in df_lags.iterrows():
#     #     dict_lags[index] = row['auto_corr']
#
#     #keep only important lads, those with autocorrelation >= 0.75
#     # threshold = 0.75
#     # important_lags = df_lags.loc[(df_lags['auto_corr'] < 1) & (df_lags['auto_corr'] >= threshold)]
#     # or:
#     # mask = (df_lags < 1) & (df_lags >= threshold)
#     # important_lags = df_lags.loc[mask.all(axis=1)]
#
#
#
#     #lag plots
#     # A lag in a time-series data is how much one point is falling behind in time from another data point. In a time-series data, data points are marked over time at varying degrees of intervals.
#     # To analyse and find out if a time-series data follows any pattern, a lag plot can be employed. A lag plot is drawn by representing the time series data in x-axis and the lag of the time series
#     # data point in y axis. For a data point, if the order of the lag is one, the lag is the previous data point. If the lag is two, the lag is data point before two data points in time.
#     # By drawing a lag plot, patterns like randomness, trends and seasonality can be searched for.
#     #manually : data['scada_lag1'] = data['scada'].shift(1)
#     #orizw nea stili shiftarismenh kai tis plotarw mazi
#
#     #using the dictionary of lags I created above and keeping only the lags with autocorr above the threshold I set
#     # threshold = 0.75
#     # for key,value in dict_lags.items():
#     #     if (value < 1) & (value >= threshold):
#     #         #important_lags = df_lags.loc[(df_lags['auto_corr'] < 1) & (df_lags['auto_corr'] >= threshold)]
#     #         data[f'cert_60m_lag{key}'] = data['cert_60m'].shift(key)
#     #         #data_na[f'cert_60m_lag{key}'] = data_na['cert_60m'].shift(key)
#     #         lag_plot = pd.plotting.lag_plot(data[f'cert_60m_lag{key}'])
#     #         lag_plot.plot(grid=True, figsize=(20, 5))
#     #         plt.title(f'Lag plot with lag={key}')
#     #         #plt.show()     needs subplots change later
#     #         plt.savefig(f'lag_plot_{key}.png')
#     #         plt.close()
#     #     else:
#     #         continue
#
#
#     # i_lags = [1,24,6]
#     # for i in i_lags:
#     #     data[f'cert_60m_lag{i}'] = data['cert_60m'].shift(i)
#     #     lag_plot = pd.plotting.lag_plot(data[f'cert_60m_lag{i}'])
#     #     lag_plot.plot(grid=True, figsize=(20, 5))
#     #     plt.title(f'Lag plot with lag={i}')
#     #     #plt.show()
#     #     plt.savefig(f'lag_plot_{i}.png')
#     #     plt.close()
#
#     # lag_plot_1 = pd.plotting.lag_plot(data['cert_60m'], lag=1)
#     # data = data.dropna().copy()
#     # lag_plot_1.plot(grid = True, figsize = (20,5))
#     # plt.title("Lag plot with lag=1")
#     # plt.savefig('lag_plot_1.png')
#     # plt.close()
#
#     # lag_plot_24 = pd.plotting.lag_plot(data['cert_60m'], lag=24)
#     # data = data.dropna().copy()
#     # lag_plot_1.plot(grid = True, figsize = (20,5))
#     # plt.title("Lag plot with lag=24")
#     # plt.savefig('lag_plot_24.png')
#     # plt.close()
#
#     # data['cert_60m_lag24'] = data['cert_60m'].shift(1,"d")      #instead of shift(24) because I have thrown 0 and nan values out
#     # data = data.dropna()
#     # lag_plot_24_v2 = pd.plotting.lag_plot(data['cert_60m_lag24'])
#     # lag_plot_24_v2.plot(grid=True, figsize=(20,5))
#     # plt.title("Lag plot with lag=24 vol2")
#     # #plt.show()
#     # plt.savefig('lag_plot_24_v2.png')
#     # plt.close()
#
#
#
#     #Seasonal decomposition
#     #seas_decomp = seasonal_decompose(data['cert_60m'], period = 24) #den xreiazetai model by default additive
#     # # print(seas_decomp.trend)
#     # # print(seas_decomp.seasonal)
#     # # print(seas_decomp.resid)
#     # # print(seas_decomp.observed)
#     seas_decomp = seasonal_decompose(data.loc[data.index[-3000]:, 'cert_60m'], period = 24)
#     plt.rcParams['figure.figsize'] = (20, 8)
#     seas_decomp.plot()
#     plt.title('Seasonal Decomposition')
#     plt.savefig('seas_decomp.png')
#     plt.close()
#
#
#     #histogram
#     #hist = data.hist()
#     data['cert_60m'].hist(bins=10, color='green', alpha=0.7)
#     plt.xlabel('cert_60m')
#     plt.ylabel('Frequency')
#     plt.title('Histogram')
#     #plt.show()
#     plt.savefig('hist.png')
#     plt.close()
#
#
#     #pearson coefficient python gia oles tis steiles -> einai gia linear corr
#     correlation_matrix = np.corrcoef(data_na, rowvar=False)
#     correlation_df = pd.DataFrame(correlation_matrix, columns=data_na.columns, index=data_na.columns)
#     correlation_df
#     #np.corrcoef(data_na['cert_60m'],data_na['SCADA_60m'])
#
#     # IQR:
#     # Interquartile Range (IQR) method is a common approach for identifying and dealing with outliers in a dataset. Outliers are data points that significantly deviate from the typical
#     # values in a dataset and can distort statistical analysis. The IQR method helps you detect and potentially remove or handle these outliers.
#     Q1 = data['cert_60m'].quantile(0.25)
#     Q3 = data['cert_60m'].quantile(0.75)
#     IQR = Q3 - Q1
#
#     lower_bound = Q1 - 0.3 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#
#     #outliers = (data['cert_60m'] < lower_bound) | (data['cert_60m'] > upper_bound)
#
#     #data[outliers] = None
#
#     # outlier = []
#     # for x in data['cert_60m']:
#     #     if ((x> upper_bound) or (x<lower_bound)):
#     #          outlier.append(x)
#     # print(' outlier in the dataset is', outlier)
#
#     res = (data_i, data_na)
#
#     return res
