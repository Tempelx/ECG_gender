#!/usr/bin/env python
__author__ = "Felix Tempel"
__copyright__ = "Copyright 2020, ECG Sex Classification"
__credits__ = ["Felix Tempel"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Felix Tempel"
__email__ = "felixtempel95@hotmail.de"
__status__ = "Production"


import matplotlib.pyplot as plt


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
