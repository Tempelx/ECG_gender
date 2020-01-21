#!/usr/bin/env python
__author__ = "Felix Tempel"
__copyright__ = "Copyright 2020, ECG Sex Classification"
__credits__ = ["Felix Tempel"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Felix Tempel"
__email__ = "felixtempel95@hotmail.de"
__status__ = "Production"

import glob
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

from biosppy.signals import ecg
from scipy.signal import medfilt

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from astropy.convolution import convolve, Box1DKernel
from src import train as trainer, feature_extraction as fe

pd.set_option('display.max_colwidth', -1)


def remove_baseline_fun(signal, signal_f_s):
    """""
    Remove the Baseline with two Median Filters.
    
    Parameters:
    ----------
    signal:     np.array ECG signal
    signal_f_s: np.array ECG sampling frequency
    
    Returns:
    ---------
    ECGblr:     np.array ECG with removed baseline
    """""

    winsize = int(round(0.2 * signal_f_s))
    if winsize % 2 == 0:
        winsize += 1
    baselineestimation = medfilt(signal, kernel_size=winsize)
    winsize = int(round(0.6 * signal_f_s))
    if winsize % 2 == 0:
        winsize += 1
    baselineestimation = medfilt(baselineestimation, kernel_size=winsize)
    ecg_blr = signal - baselineestimation

    return ecg_blr


def plot_signals(ecgblr=None,  convolved_data_7_5=None,
                 signal_data=None, idx_df=None):

    # plot signals
    plt.figure()
    plt.plot(ecgblr)
    plt.plot(convolved_data_7_5)
    plt.plot(signal_data)
    # R-Peaks

    # r_peaks = ecg_object["filtered"][idx_df["R-Peak"].values.tolist()]
    r_peaks = convolved_data_7_5[idx_df["R-Peak"].values.tolist()]
    plt.plot(idx_df["R-Peak"].values.tolist(), r_peaks, 'r*')

    # S-Peak
    # s_peaks = ecg_object["filtered"][idx_df["S-Peak"].values.tolist()]
    s_peaks = convolved_data_7_5[idx_df["S-Peak"].values.tolist()]

    plt.plot(idx_df["S-Peak"].values.tolist(), s_peaks, 'b*')

    # J-Point
    # j_point = ecg_object["filtered"][idx_df["J-Point"].values.tolist()]
    j_point = convolved_data_7_5[idx_df["J-Point"].values.tolist()]
    plt.plot(idx_df["J-Point"].values.tolist(), j_point, 'k*')

    # T-Peak
    # t_peaks = ecg_object["filtered"][idx_df["T-Peak"].values.tolist()]
    t_peaks = convolved_data_7_5[idx_df["T-Peak"].values.tolist()]

    plt.plot(idx_df["T-Peak"].values.tolist(), t_peaks, 'y*')
    plt.legend(["Baseline removed", "Convolved_7_5", "RAW", "R-Peak",
                "S-Peak", "J-Point", "T-Peak"])
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()


def idx_finder_2(ecg_object, filtered_signal, signal_number):
    # TODO: auswahl eines anderen leads?...
    idx_args = np.zeros((int(ecg_object['rpeaks'].size - 1), 3))
    idx_count = 0
    area_s_search = 40
    area_j_search = 40
    for i in ecg_object['rpeaks'][:-1]:

        area_s = filtered_signal[i: i + area_s_search]

        # find S Point in V2 take first
        try:
            idx_s_peaks, _ = scipy.signal.find_peaks(-area_s)
            idx_s = idx_s_peaks[0]
        except ValueError:
            # try other lead
            idx_s = None

        area_j = filtered_signal[i + idx_s: i + idx_s + area_j_search]

        slope_diff = np.diff(area_j)
        area_j_slope = np.where(slope_diff < 0)

        time_var = 13
        try:
            idx_diff_tp_time = area_j_slope[0][np.min(np.where(area_j_slope[0] >= time_var))]
            idx_j = idx_s + idx_diff_tp_time
        except ValueError:
            idx_j = idx_s + 40

        idx_args[idx_count] = [i, i + idx_s, i + idx_j]
        idx_count += 1

    idx_data = pd.DataFrame(idx_args, columns=["R-Peak", "S-Peak", "J-Point"], dtype=np.int16)

    return idx_data


def idx_finder_t(r_peaks, filtered_signal, idx_df):

    # 30 ms search area for T
    # 0.20 to 0.40ms for QT
    search_area = 70
    idx_count = 0
    idx_arg = np.zeros((int(r_peaks.size - 1), 1))

    for idx, val in enumerate(r_peaks[:-1]):
        if idx == r_peaks.size:
            break

        area_t = filtered_signal[val + search_area: r_peaks[idx+1] - search_area]

        try:
            idx_arg[idx_count] = np.argmax(area_t) + val + search_area
        except ValueError:
            idx_arg[idx_count] = None

        idx_count += 1

    if idx_df.shape[0] != idx_arg.size:
        breakpoint()

    idx_df["T-Peak"] = idx_arg.astype(int)

    return idx_df


def lead_search_fun(signal_data):
    """""
    Calculate Lead with highest T-Peak

    Parameters:
    ----------
    signal_data:    np.array    ECG signal 

    Returns:
    ---------
    lead:           int         highest Lead
    """""

    t_value = np.zeros(6)
    counter = 0
    leads = [0, 1, 8, 9, 10, 11]
    # colour = ['r*', 'g*', 'b*', 'y*', 'k*', 'c*']
    # plt.figure(1)
    for i in leads:
        # next high point after S-Peak
        try:
            ecg_lead = ecg.ecg(signal_data[i, 0:1500], sampling_rate=500, show=False)
            search_area_blr = remove_baseline_fun(signal_data[i, 0:1500], 500)
            s_index = np.argmin(search_area_blr[ecg_lead["rpeaks"][0]: ecg_lead["rpeaks"][0] + 50]) + ecg_lead["rpeaks"][0]
            t_index = np.argmax(search_area_blr[s_index: ecg_lead["rpeaks"][1] - 20]) + s_index
            t_value[counter] = search_area_blr[t_index+5]
        except ValueError:
            t_value[counter] = 0

        counter += 1

    return leads[np.argmax(t_value)]


def main():
    print('Start Feature extraction...')

    # Chinese database
    ecg_files_chinese = sorted(glob.glob('/Users/felixtempel11/Documents/database/chin_database/*.mat'))
    ecg_files_chinese_ref = '/Users/felixtempel11/Documents/database/chin_database/REFERENCE.csv'

    ref_data = pd.read_csv(ecg_files_chinese_ref)

    # Feature Dataframe
    df_col = ["Meas_no", "Sex", "Age", "st_angle_mean", "st_angle_std", "JT_mean", "JT_std", "T_asc_mean",
              "T_asc_std", "T_desc_mean", "T_desc_std"]

    feature_data = np.zeros((ref_data["First_label"].eq(1).sum(), df_col.__len__()))
    feature_count = 0
    from progress.bar import Bar

    bar = Bar('Processing Features', max=len(ecg_files_chinese))

    for i in range(0, len(ecg_files_chinese)):

        signal_number = ecg_files_chinese[i][-8:-4]

        # only load normal rhythms --> first_label == 1
        if ref_data.loc[[i], ['First_label']].values[0][0] == 1:

            signal_raw = scipy.io.loadmat(ecg_files_chinese[i])

            # extract data from mat-file
            signal_sex = signal_raw['ECG']['sex'][0][0][0][0]
            if signal_sex == 'F':
                signal_sex = 1
            elif signal_sex == 'M':
                signal_sex = 0

            signal_age = signal_raw['ECG']['age'][0][0][0][0]
            signal_data = signal_raw['ECG']['data'][0][0]
            signal_f_s = 500.0  # Hz

            # search for highest T-Wave in Lead 1,2 and V3-V6
            # first 5sec for computation
            lead_t_highest = lead_search_fun(signal_data)
            lead_j = 7

            # Leads [1, 2, 3, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
            # get indices from chosen lead
            ecg_object = ecg.ecg(signal=signal_data[lead_j, :], sampling_rate=500, show=False)

            # Custom  filters
            # Baseline remove
            ecg_blr = remove_baseline_fun(signal=signal_data[lead_j, :], signal_f_s=signal_f_s)

            # Box Filter
            # 7.5 seems to be the best...
            ecg_conv = convolve(ecg_blr, kernel=Box1DKernel(7.5))

            idx_df = idx_finder_2(ecg_object, ecg_blr, signal_number)

            # T wave detection
            ecg_object_t = ecg.ecg(signal=signal_data[lead_t_highest, :], sampling_rate=500, show=False)
            ecg_blr_t = remove_baseline_fun(signal=signal_data[lead_t_highest, :], signal_f_s=signal_f_s)
            ecg_conv_t = convolve(ecg_blr_t, kernel=Box1DKernel(7.5))

            if ecg_object_t['rpeaks'].size < ecg_object['rpeaks'].size:
                # drop last R-Peak in ecg
                if abs(ecg_object_t['rpeaks'][0] - ecg_object['rpeaks'][0]) > 25:
                    # earlier R-Peak at beginning
                    r_peaks = ecg_object_t['rpeaks'][1:-1]
                else:
                    r_peaks = ecg_object['rpeaks'][:-1]
                    idx_df = idx_df.head(-1)
            elif ecg_object_t['rpeaks'].size > ecg_object['rpeaks'].size:
                # drop last R-Peak in ecg_t
                if abs(ecg_object_t['rpeaks'][0] - ecg_object['rpeaks'][0]) > 25:
                    # earlier R-Peak at beginning
                    r_peaks = ecg_object_t['rpeaks'][1:-1]
                    idx_df = idx_df.head(-1)
                else:
                    r_peaks = ecg_object_t['rpeaks'][:-1]

            else:
                r_peaks = ecg_object_t['rpeaks']

            idx_df = idx_finder_t(r_peaks, ecg_blr_t, idx_df)

            # check plausibility of DataFrame
            if any(idx_df['T-Peak'] - idx_df['J-Point'] <= 0):
                idx_df = idx_df.drop(idx_df.loc[idx_df['T-Peak'] - idx_df['J-Point'] <= 0].index[0])

            if idx_df.empty:
                breakpoint()
            # find indices from Lead 1
            # idx_df = idx_finder_fun(ecg_object, ECGblr, signal_data[lead_highest, :])

            # plot_signals(ecgblr=ECGblr, convolved_data_7_5=ECGconv,
            #             signal_data=signal_data[lead_t_highest, :], idx_df=idx_df)

            # Calculate features
            st_angle = fe.st_degree_fun(ecg_conv, idx_df)
            jt_max_ms = fe.jt_max_fun(idx_df, signal_f_s)
            t_asc_degree = fe.t_asc_fun(ecg_conv_t, idx_df)
            t_des_degree = fe.t_des_fun(ecg_conv_t, idx_df)

            df_data = [signal_number, signal_sex, signal_age, st_angle[1], st_angle[2], jt_max_ms[1], jt_max_ms[2],
                       t_asc_degree[1], t_asc_degree[2], t_des_degree[1], t_des_degree[2]]

            feature_data[feature_count] = df_data
            feature_count += 1
            bar.next()

    bar.finish()

    # Feature DataFrame
    features = pd.DataFrame(feature_data, columns=df_col)
    features.to_csv('/Users/felixtempel11/PycharmProjects/ECG_sex_estimation/data/features.csv')

    # mask value which are above std threshold
    mask_st = features.st_angle_std.index[features['st_angle_std'] > 10].tolist()
    mask_t_asc = features.st_angle_std.index[features['T_asc_std'] > 10].tolist()
    mask_t_des = features.st_angle_std.index[features['T_desc_std'] > 10].tolist()

    # mask outliners
    mask_st_2 = features.st_angle_std.index[features['st_angle_mean'] > 5].tolist()
    mask_t_asc_2 = features.st_angle_std.index[features['T_asc_mean'] > 5].tolist()
    mask_t_des_2 = features.st_angle_std.index[features['T_desc_mean'] > 5].tolist()

    features.loc[mask_st, 'st_angle_mean'] = np.nan
    features.loc[mask_st, 'st_angle_std'] = np.nan
    features.loc[mask_st_2, 'st_angle_mean'] = np.nan
    features.loc[mask_st_2, 'st_angle_std'] = np.nan
    features.loc[mask_t_asc, 'T_asc_mean'] = np.nan
    features.loc[mask_t_asc, 'T_asc_std'] = np.nan
    features.loc[mask_t_asc_2, 'T_asc_mean'] = np.nan
    features.loc[mask_t_asc_2, 'T_asc_std'] = np.nan
    features.loc[mask_t_des, 'T_desc_mean'] = np.nan
    features.loc[mask_t_des, 'T_desc_std'] = np.nan
    features.loc[mask_t_des_2, 'T_desc_mean'] = np.nan
    features.loc[mask_t_des_2, 'T_desc_std'] = np.nan


    min_max_scaler = preprocessing.MinMaxScaler()
    std_scaler = preprocessing.StandardScaler()

    df_minmax = min_max_scaler.fit_transform(features[['st_angle_mean', 'JT_mean', 'T_asc_mean', 'T_desc_mean']])
    df_std_scaler = std_scaler.fit_transform(features[['st_angle_mean', 'JT_mean', 'T_asc_mean', 'T_desc_mean']])

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_minmax = imp_mean.fit_transform(df_minmax)
    df_std_scaler = imp_mean.fit_transform(df_std_scaler)


    scaled_minmax = np.concatenate((feature_data[:, 0:3], df_minmax), axis=1)
    scaled_std = np.concatenate((feature_data[:, 0:3], df_std_scaler), axis=1)

    df_col_scaled = ["Meas_no", "Sex", "Age", "ST_angle_scaled", "JT_scaled", "T_asc_scaled", "T_desc_scaled"]

    features_scaled_minmax = pd.DataFrame(scaled_minmax, columns=df_col_scaled)
    features_scaled_std = pd.DataFrame(scaled_std, columns=df_col_scaled)

    X_minmax = features_scaled_minmax[features_scaled_minmax.columns[3:9]].values
    X_std = features_scaled_std[features_scaled_minmax.columns[3:9]].values
    y = features_scaled_minmax.iloc[:,1].values


    # TODO: train model
    trainer.train_decision_tree(X_minmax,y)
    trainer.train_svm(X_minmax, y)
    trainer.train_xgboost(X_minmax, y)
    trainer.train_svm(X_minmax, y)
    trainer.train_random_forest(X_minmax, y)
    trainer.train_log_reg(X_minmax, y)
    trainer.train_adaboost(X_minmax, y)
    trainer.train_knn(X_minmax, y)
    trainer.train_mlp(X_minmax, y)
    trainer.train_qda(X_minmax, y)


    trainer.train_decision_tree(X_std, y)
    trainer.train_svm(X_std, y)
    trainer.train_xgboost(X_std, y)
    trainer.train_svm(X_std, y)
    trainer.train_random_forest(X_std, y)
    trainer.train_log_reg(X_std, y)
    trainer.train_adaboost(X_std, y)
    trainer.train_knn(X_std, y)
    trainer.train_mlp(X_std, y)
    trainer.train_qda(X_std, y)


if __name__ == '__main__':

    main()
