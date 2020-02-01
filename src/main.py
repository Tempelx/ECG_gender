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

from biosppy.signals import ecg

from progress.bar import Bar
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from astropy.convolution import convolve, Box1DKernel
from src import train as trainer, feature_extraction as fe, preprocessing as pp


def feature_calc():
    print('Start Feature extraction...')

    # Chinese database
    ecg_files_chinese = sorted(glob.glob('/Users/felixtempel11/Documents/database/chin_database/*.mat'))
    ecg_files_chinese_ref = '/Users/felixtempel11/Documents/database/chin_database/REFERENCE.csv'

    ref_data = pd.read_csv(ecg_files_chinese_ref)

    # Feature Dataframe
    df_col = ["Meas_no", "Sex", "Age", "st_angle_mean", "st_angle_std", "JT_mean", "JT_std", "T_asc_mean",
              "T_asc_std", "T_desc_mean", "T_desc_std", "T_max_mean", "T_max_std"]

    feature_data = np.zeros((ref_data["First_label"].eq(1).sum(), df_col.__len__()))
    feature_count = 0

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
            lead_t_highest = pp.lead_search_fun(signal_data)
            lead_j = 7

            # Leads [1, 2, 3, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
            # get indices from chosen lead
            ecg_object = ecg.ecg(signal=signal_data[lead_j, :], sampling_rate=500, show=False)

            # Custom  filters
            # Baseline remove
            ecg_blr = pp.remove_baseline_fun(signal=signal_data[lead_j, :], signal_f_s=signal_f_s)

            # Box Filter
            # 7.5 seems to be the best...
            ecg_conv = convolve(ecg_blr, kernel=Box1DKernel(7.5))

            idx_df = pp.idx_finder(ecg_object, ecg_blr, signal_number)

            # T wave detection
            ecg_object_t = ecg.ecg(signal=signal_data[lead_t_highest, :], sampling_rate=500, show=False)
            ecg_blr_t = pp.remove_baseline_fun(signal=signal_data[lead_t_highest, :], signal_f_s=signal_f_s)
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

            idx_df = pp.idx_finder_t(r_peaks, ecg_blr_t, idx_df)

            # check plausibility of DataFrame
            if any(idx_df['T-Peak'] - idx_df['J-Point'] <= 0):
                idx_df = idx_df.drop(idx_df.loc[idx_df['T-Peak'] - idx_df['J-Point'] <= 0].index[0])

            # Calculate features
            st_angle = fe.st_degree_fun(ecg_conv, idx_df)
            jt_max_ms = fe.jt_max_fun(idx_df, signal_f_s)
            t_asc_degree = fe.t_asc_fun(ecg_conv_t, idx_df)
            t_des_degree = fe.t_des_fun(ecg_conv_t, idx_df)
            t_max = fe.t_ampl_fun(ecg_conv_t, idx_df)

            df_data = [signal_number, signal_sex, signal_age, st_angle[1], st_angle[2], jt_max_ms[1], jt_max_ms[2],
                       t_asc_degree[1], t_asc_degree[2], t_des_degree[1], t_des_degree[2], t_max[1], t_max[2]]

            feature_data[feature_count] = df_data
            feature_count += 1
        bar.next()
    bar.finish()

    # Feature DataFrame
    feat = pd.DataFrame(feature_data, columns=df_col)
    feat.to_csv('/Users/felixtempel11/PycharmProjects/ECG_sex_estimation/data/features.csv')

    return feat


def train_features(feature_path):

    print('Start Training...')
    features = pd.read_csv(feature_path, index_col=0)

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

    # Scale features
    min_max_scaler = preprocessing.MinMaxScaler()
    std_scaler = preprocessing.StandardScaler()

    df_minmax = min_max_scaler.fit_transform(features[['st_angle_mean', 'JT_mean', 'T_asc_mean', 'T_desc_mean',
                                                       'T_max_mean']])
    df_std_scaler = std_scaler.fit_transform(features[['st_angle_mean', 'JT_mean', 'T_asc_mean', 'T_desc_mean',
                                                       'T_max_mean']])

    # Imputer for NaN values
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_minmax = imp_mean.fit_transform(df_minmax)
    df_std_scaler = imp_mean.fit_transform(df_std_scaler)

    feature_dummy = features.iloc[:, 0:3].to_numpy()

    scaled_minmax = np.concatenate((feature_dummy, df_minmax), axis=1)
    scaled_std = np.concatenate((feature_dummy, df_std_scaler), axis=1)

    df_col_scaled = ["Meas_no", "Sex", "Age", "ST_angle_scaled", "JT_scaled", "T_asc_scaled", "T_desc_scaled",
                     "T_max_scaled"]

    features_scaled_minmax = pd.DataFrame(scaled_minmax, columns=df_col_scaled)
    features_scaled_std = pd.DataFrame(scaled_std, columns=df_col_scaled)

    X_minmax = features_scaled_minmax[features_scaled_minmax.columns[3:9]].values
    X_std = features_scaled_std[features_scaled_minmax.columns[3:9]].values
    y = features_scaled_minmax.iloc[:, 1].values

    i = 0
    print('Train full dataset')
    trainer.train_classifiers(X_minmax, y, 'Full Dataset', i)
    print('Finished full dataset')

    feat_low_age = features_scaled_minmax.loc[features_scaled_minmax['Age'] <= 30]
    feat_low_scaled = feat_low_age[feat_low_age.columns[3:9]].values
    y = feat_low_age.iloc[:, 1].values

    i = 1
    print('Train low age ')
    trainer.train_classifiers(feat_low_scaled, y, ' Age <= 30 ', i)
    print('Finished low age')

    feat_high_age = features_scaled_minmax.loc[features_scaled_minmax['Age'] >= 30]
    feat__high_scaled = feat_high_age[feat_high_age.columns[3:9]].values
    y = feat_high_age.iloc[:, 1].values

    i = 2
    print('Train high age')
    trainer.train_classifiers(feat__high_scaled, y, 'Age >= 30', i)
    print('Finished high age')

    import matplotlib.pyplot as plt
    plt.show()
    input()
    # test


if __name__ == '__main__':

    #features = feature_calc()
    feature_path = '/Users/felixtempel11/PycharmProjects/ECG_sex_estimation/data/features.csv'
    train_features(feature_path)

