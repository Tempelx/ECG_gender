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
import matplotlib.pyplot as plt

def st_degree_fun(signal_conv, idx_df):
    """""
    Calculate ST angle.

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
        # J-Point + 60 msec == 300 samples --> far too wide!!!
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

    # TODO: Crosscheck with real data?!
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
    Calculate steepest point between J-Point and T-Peak,
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
        # steepest_idx = np.argmax(area[:-1] - area[1:])
        # steepest_val = np.max(area[:-1] - area[1:])

        # TODO: crosscheck for measurement local min --> diff?
        # TODO: baseline really at 0??????

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
    Calculate steepest point after T-Peak,
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
        # y = mx+b

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
        #angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        counter += 1

    # mean + std
    t_des_mean = np.nanmean(t_des_angle)
    t_des_std = np.nanstd(t_des_angle)

    return t_des_angle, t_des_mean, t_des_std


def t_ampl_fun(signal, t_peak_idx_df, column):
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

    # get Amplitude of T-Wave wrt. baseline
    counter = 0
    t_ampl = np.zeros(t_peak_idx_df[column].size)

    for i in t_peak_idx_df[column]:
        t_ampl[counter] = signal[int(i)]
        counter += 1

    t_ampl_mean = np.mean(t_ampl)
    t_ampl_std = np.std(t_ampl)

    return t_ampl, t_ampl_mean, t_ampl_std
