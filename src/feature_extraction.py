#!/usr/bin/env python
__author__ = "Felix Tempel"
__copyright__ = "Copyright 2020, ECG Sex Classification"
__credits__ = ["Felix Tempel"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Felix Tempel"
__email__ = "felixtempel95@hotmail.de"
__status__ = "Production"

import numpy as np
import pandas as pd
import glob
import scipy
from biosppy.signals import ecg
from astropy.convolution import convolve, Box1DKernel
from src import preprocessing as pp


def feature_calc(path):
    """""
    Main loop function for feature calculation

    Parameters:
    ----------
    path      String           path to chinese database on local machine

    Returns:
    ---------
    feat:      pd.DataFrame    Calculated Features 
    """""
    print('Start Feature extraction...')

    # Chinese database
    ecg_files_chinese = sorted(glob.glob(path+'*.mat'))
    ecg_files_chinese_ref = './data/REFERENCE.csv'

    ref_data = pd.read_csv(ecg_files_chinese_ref)

    # Feature Dataframe
    df_col = ["Meas_no", "Sex", "Age", "st_angle_mean", "st_angle_std", "JT_mean", "JT_std", "T_asc_mean",
              "T_asc_std", "T_desc_mean", "T_desc_std", "T_max_mean", "T_max_std"]

    feature_data = np.zeros((ref_data["First_label"].eq(1).sum(), df_col.__len__()))
    feature_count = 0

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
            lead_t_highest = pp.lead_search_fun(signal_data)
            lead_j = 7

            # Leads [1, 2, 3, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
            # get indices from chosen lead
            ecg_object = ecg.ecg(signal=signal_data[lead_j, :], sampling_rate=signal_f_s, show=False)

            # Custom  filters
            # Baseline remove
            ecg_blr = pp.remove_baseline_fun(signal=signal_data[lead_j, :], signal_f_s=signal_f_s)

            # Box Filter
            # 7.5 seems to be the best...
            ecg_conv = convolve(ecg_blr, kernel=Box1DKernel(7.5))

            idx_df = pp.idx_finder(ecg_object, ecg_blr, signal_number)

            # T wave detection
            ecg_object_t = ecg.ecg(signal=signal_data[lead_t_highest, :], sampling_rate=signal_f_s, show=False)
            ecg_blr_t = pp.remove_baseline_fun(signal=signal_data[lead_t_highest, :], signal_f_s=signal_f_s)
            ecg_conv_t = convolve(ecg_blr_t, kernel=Box1DKernel(7.5))

            # check if both ecg_objects have the same number of R-Peaks
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

            idx_df = pp.idx_finder_t(r_peaks, ecg_blr_t, idx_df)

            # check plausibility of DataFrame
            if any(idx_df['T-Peak'] - idx_df['J-Point'] <= 0):
                idx_df = idx_df.drop(idx_df.loc[idx_df['T-Peak'] - idx_df['J-Point'] <= 0].index[0])

            # Calculate features
            st_angle = st_degree_fun(ecg_conv, idx_df)
            jt_max_ms = jt_max_fun(idx_df, signal_f_s)
            t_asc_degree = t_asc_fun(ecg_conv_t, idx_df)
            t_des_degree = t_des_fun(ecg_conv_t, idx_df)
            t_max = t_ampl_fun(ecg_conv_t, idx_df)

            df_data = [signal_number, signal_sex, signal_age, st_angle[1], st_angle[2], jt_max_ms[1], jt_max_ms[2],
                       t_asc_degree[1], t_asc_degree[2], t_des_degree[1], t_des_degree[2], t_max[1], t_max[2]]

            feature_data[feature_count] = df_data
            feature_count += 1

    # Feature DataFrame
    feat = pd.DataFrame(feature_data, columns=df_col)
    # save this stuff
    feat.to_csv('./data/features.csv')

    return feat


def st_degree_fun(signal_conv, idx_df):
    """""
    Calculate ST angle via gradient angle.

    Parameters:
    ----------
    signal_conv:    np.array ECG signal convolved
    idx_df:         pd.DataFrame DataFrame with indices

    Returns:
    ---------
    st_angle, st_mean, st_std:     np.array ST angle with std and mean
    """""

    st_angle = np.zeros((idx_df["J-Point"].size, 1))
    counter = 0
    for i in idx_df["J-Point"]:
        # J-Point + 50 samples
        v1 = [i, signal_conv[i]]
        v2 = [i + 50, signal_conv[i + 50]]

        m = (v2[1] - v1[1]) / (v2[0] - v1[0])
        b = v1[1] - m * v1[0]

        zero_crossing = np.round(0 - b / m)

        x1 = [(zero_crossing + 1) - zero_crossing, 0]
        x2 = [v2[0] - zero_crossing, v2[1]]

        cosine_angle = np.arccos(np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2)))
        st_angle[counter] = np.rad2deg(cosine_angle)

        counter += 1

    # mean + std
    st_mean = np.mean(st_angle)
    st_std = np.std(st_angle)

    return st_angle, st_mean, st_std


def jt_max_fun(idx_df, signal_f_s):
    """""
    Calculate distance from J-Point to T-Peak in ms.

    Parameters:
    ----------
    signal_f_s:     float        ECG sampling frequency
    idx_df:         pd.DataFrame DataFrame with indices

    Returns:
    ---------
    distance_ms, distance_mean, distance_std:     np.array distances with std and mean
    """""

    distance_ms = np.zeros(idx_df["J-Point"].size)
    counter = 0
    for i, ii in zip(idx_df["J-Point"], idx_df["T-Peak"]):
        distance_samples = ii - i
        distance_ms[counter] = distance_samples * (1 / signal_f_s)
        counter += 1

    distance_mean = np.mean(distance_ms)
    distance_std = np.std(distance_ms)

    return distance_ms, distance_mean, distance_std


def t_asc_fun(signal_conv, idx_df):
    """""
    Calculate steepest ascending point between J-Point and T-Peak,
    then calculate the angle with respect to baseline.

    Parameters:
    ----------
    signal_conv:    np.array        ECG signal convolved
    idx_df:         pd.DataFrame    DataFrame with indices

    Returns:
    ---------
    t_asc_angle, t_asc_mean, t_asc_std:    np.array    T ascending angle with std and mean
    """""

    counter = 0
    t_asc_angle = np.zeros(idx_df["T-Peak"].size)

    for i, ii in zip(idx_df["J-Point"], idx_df["T-Peak"]):

        idx_start = i
        idx_end = ii
        area = signal_conv[idx_start:idx_end]

        if area.size == 0:
            breakpoint()

        steepest_idx = np.argmax(np.gradient(area))
        steepest_val = area[steepest_idx]

        # angle over two vectors
        try:
            v1 = [steepest_idx, steepest_val]
            v2 = [steepest_idx + 1, area[steepest_idx+1]]
            m = (v2[1] - v1[1]) / (v2[0] - v1[0])
            b = v1[1] - m * v1[0]

            zero_crossing = np.round(0 - b / m)

            x1 = [(zero_crossing + 1) - zero_crossing, 0]
            x2 = [v1[0] - zero_crossing, v1[1]]

            cosine_angle = np.arccos(np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2)))
            t_asc_angle[counter] = np.rad2deg(cosine_angle)
        except IndexError:
            t_asc_angle[counter] = np.nan

        counter += 1

    # mean + std
    t_asc_mean = np.nanmean(t_asc_angle)
    t_asc_std = np.nanstd(t_asc_angle)

    return t_asc_angle, t_asc_mean, t_asc_std


def t_des_fun(signal_conv, idx_df):
    """""
    Calculate steepest descending point after T-Peak,
    then calculate the angle with respect to baseline.

    Parameters:
    ----------
    signal_conv:    np.array        ECG signal convolved
    idx_df:         pd.DataFrame    DataFrame with indices

    Returns:
    ---------
    t_des_angle, t_des_mean, t_des_std:    np.array    T descending angle with std and mean
    """""

    counter = 0
    t_des_angle = np.zeros(idx_df["T-Peak"].size)

    for i in idx_df["T-Peak"]:
        idx_start = i
        idx_end = i + 35
        area = signal_conv[idx_start:idx_end]

        steepest_idx = np.argmin(np.gradient(area))
        steepest_val = area[steepest_idx]

        # crossing with baseline
        # angle over two vectors
        try:
            v1 = [steepest_idx, steepest_val]
            v2 = [steepest_idx + 1, area[steepest_idx + 1]]

            m = (v1[1] - v2[1]) / (v1[0] - v2[0])
            b = v1[1] - m * v1[0]

            zero_crossing = np.round(0 - b / m)

            x1 = [(zero_crossing - 1) - zero_crossing, 0]
            x2 = [v1[0] - zero_crossing, v1[1]]

            cosine_angle = np.arccos(np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2)))
            t_des_angle[counter] = np.rad2deg(cosine_angle)
        except IndexError:
            t_des_angle[counter] = np.nan

        counter += 1

    # mean + std
    t_des_mean = np.nanmean(t_des_angle)
    t_des_std = np.nanstd(t_des_angle)

    return t_des_angle, t_des_mean, t_des_std


def t_ampl_fun(signal_conv, idx_df):
    """""
    Calculate Amplitude of T-Wave wrt. baseline

    Parameters:
    ----------
    signal:    np.array        ECG signal 
    idx_df:    pd.DataFrame    DataFrame with indices
    column:    int             column 

    Returns:
    ---------
    t_ampl, t_ampl_mean, t_ampl_std:    np.array    T amplitude with std and mean
    """""

    counter = 0
    t_ampl = np.zeros(idx_df["T-Peak"].size)
    for i in idx_df["T-Peak"]:
        t_ampl[counter] = signal_conv[int(i)]
        counter += 1

    t_ampl_mean = np.nanmean(t_ampl)
    t_ampl_std = np.nanstd(t_ampl)

    return t_ampl, t_ampl_mean, t_ampl_std
