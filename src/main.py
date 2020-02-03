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
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from src import train as trainer, feature_extraction as fe


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

    df_minmax = min_max_scaler.fit_transform(features[['st_angle_mean', 'JT_mean', 'T_asc_mean', 'T_desc_mean',
                                                       'T_max_mean']])

    # Imputer for NaN values
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_minmax = imp_mean.fit_transform(df_minmax)

    feature_dummy = features.iloc[:, 0:3].to_numpy()

    scaled_minmax = np.concatenate((feature_dummy, df_minmax), axis=1)

    df_col_scaled = ["Meas_no", "Sex", "Age", "ST_angle_scaled", "JT_scaled", "T_asc_scaled", "T_desc_scaled",
                     "T_max_scaled"]

    features_scaled_minmax = pd.DataFrame(scaled_minmax, columns=df_col_scaled)

    X_minmax = features_scaled_minmax[features_scaled_minmax.columns[3:9]].values
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

    # path to chinese database files on your local machine
    # path_to_chin = ''
    # features = fe.feature_calc(path_to_chin)

    #path_to_nsr = '/Users/felixtempel11/Documents/database/nsrdb/'
    #features = fe.feature_calc_nsr(path_to_nsr)

    feature_path = './data/features.csv'
    train_features(feature_path)

