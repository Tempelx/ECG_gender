#!/usr/bin/env python
__author__ = "Felix Tempel"
__copyright__ = "Copyright 2020, ECG Sex Classification"
__credits__ = ["Felix Tempel"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Felix Tempel"
__email__ = "felixtempel95@hotmail.de"
__status__ = "Production"

from scipy.signal import medfilt
import numpy as np
import scipy


def idx_finder(ecg_object, filtered_signal, signal_number):

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


def remove_baseline_fun(signal, signal_f_s):
    """""
    Remove the Baseline with two Median Filters.

    Parameters:
    ----------
    signal:     np.array ECG signal
    signal_f_s: np.array ECG sampling frequency

    Returns:
    ---------
    ecg_blr:     np.array ECG with removed baseline
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
