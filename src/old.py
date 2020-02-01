import pandas as pd
import numpy as np
import wfdb
import glob

def dat_to_csv():
    print('Convert Database to csv...')
    # Get list of all .dat files in the current folder
    dat_files = glob.glob('/Users/felixtempel11/Documents/database/apnea-ecg/*.dat')
    df = pd.DataFrame(data=dat_files)
    # Write the list to a CSV file
    df.to_csv("files_list.csv", index=False, header=None, sep=",")
    files = pd.read_csv("files_list.csv", header=None)
    for i in range(1, len(files) + 1):
        recordname = str(files.iloc[[i]])
        print(recordname)
        print(recordname[:-7])
        recordname_new = recordname[-7:-4]
        print(recordname_new)
        # Extracting just the filename part (will differ from database to database)
        record = wfdb.rdsamp('/Users/felixtempel11/Documents/database/apnea-ecg/' + recordname_new)
        # rdsamp() returns the signal as a numpy array

        record = np.asarray(record[0])
        path = recordname_new + ".csv"
        # Writing the CSV for each record
        np.savetxt(path, record, delimiter=",")

        print("Files done: %s/%s" % (i, len(files)))

    print("\nAll files done!")

def read_data():





    mat2 = scipy.io.loadmat('/Users/felixtempel11/Downloads/TrainingSet1/A1981.mat')
    ecg_object = ecg.ecg(signal=signal, sampling_rate=500, show=True)

    ecg_files_apnea = glob.glob('/Users/felixtempel11/Documents/database/apnea-ecg/*.dat')
    # normal sinus
    ecg_files_nsr   = glob.glob('/Users/felixtempel11/Documents/database/nsrdb/*.dat')



    """""
    nsr_data   = ['16265', '16272',	'16273', '16420', '16483', '16539', '16773', '16786', '16795', '17052',	'17453',
                  '18177',	'18184',	'19088',	'19090',	'19093',	'19140',	'19830']
    apnea_data = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 'a11', 'a12', 'a13', 'a14',
                  'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'b01', 'b02', 'b03', 'b05', 'b04', 'c01', 'c02', 'c03',
                  'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10']
    """""
    apnea_sex  = []

    apnea_df = pd.DataFrame(columns=["Measurement", "Age", "Sex", "feature"])
    nsr_df   = pd.DataFrame(columns=["Measurement", "Age", "Sex", "feature"])

    # load apnea database

    list_meas = []
    list_sex = []
    list_age = []
    for i in range(0, len(ecg_files_nsr)):
        record = wfdb.rdrecord(record_name=ecg_files_nsr[i][:-4])
        age_sex = record.comments[0].split()
        list_age.append(age_sex[0])
        list_sex.append(age_sex[1])

        list_meas.append(record.record_name)



    data = np.transpose(np.asarray([list_meas, list_age, list_sex]))
    list_sex = pd.DataFrame(data, columns=["Measurement", "Age", "Sex"])




    from sklearn import preprocessing
    for i in range(0, len(ecg_files_apnea)):

        record, fields = wfdb.rdsamp(ecg_files_nsr[i][:-4])
        #wfdb.plot_wfdb(record)
        #print(record.__dict__)
        # Signal length in secon
        signal_T = (fields.get("sig_len") / fields.get("fs") )
        # cut to 20 sec
        signal_range = fields.get("fs") * 20
        signal_t = np.arange(0, signal_range, 1 / fields.get("fs"))
        # scale signal
        signal_y = preprocessing.scale(np.nan_to_num(record[0:signal_range, 0]))

        ecg_object = ecg.ecg(signal=signal_y, sampling_rate=fields.get("fs"), show=True)
        #ecg_object_2 = ecg.ecg(signal=record[0:signal_range, 0], sampling_rate=fields.get("fs"), show=True)
        # append Features to Dataframe



def filter():
    # compare to own filter
    # Sample rate and desired cutoff frequencies (in Hz).

    order = 3
    filtered, _, _ = st.filter_signal(signal=signal_data[j, :],
                                      ftype='butter',
                                      band='bandpass',
                                      order=order,
                                      frequency=[0.5, 10],
                                      sampling_rate=signal_f_s)

    plt.figure()
    plt.plot(filtered)
    j_point2 = filtered[idx_df["J-Point"].values.tolist()]

    plt.plot(idx_df["J-Point"].values.tolist(), j_point2, 'k*')

    lowcut = 0.5
    highcut = 10.0

    t_signal = len(signal_data[1]) / signal_f_s
    time = np.arange(0, t_signal, 1 / signal_f_s)

    b, a = signal.butter(2, [lowcut, highcut], 'bandpass', fs=500)
    signal_filtered = signal.filtfilt(b, a, signal_data[0, :])

    plt.figure()
    plt.plot(signal_filtered)

def rdp_algo():
    x = np.arange(0, 80, 1)
    traj = np.column_stack((x, area_of_interest))

    simplified_trajectory = rdp(traj, epsilon=0.05)
    sx, sy = simplified_trajectory.T

    # Visualize trajectory and its simplified version.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, area_of_interest, 'r--', label='trajectory')
    ax.plot(sx, sy, 'b-', label='simplified trajectory')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc='best')

    # find turning points

    min_angle = np.pi / 2000.0
    theta = angle(simplified_trajectory)

    # Select the index of the points with the greatest theta.
    # Large theta is associated with greatest change in direction.
    idx = np.where(theta > min_angle)[0] + 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sx, sy, 'gx-', label='simplified trajectory')
    ax.plot(sx[idx], sy[idx], 'ro', markersize=7, label='turning points')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc='best')

    import pywt
    import numpy as np

    f_s_signal = 500  # Hz
    T_signal = len(signal_data[1]) / f_s_signal
    time = np.arange(0, T_signal, 1 / f_s_signal)
    dt = time[1] - time[0]

    # Calculate continuous wavelet transform
    coef, freqs = pywt.cwt(x, np.arange(1, 50), 'morl',
                           sampling_period=sampling_period)

    scales = np.arange(1, 500)

    [cfs, frequencies] = pywt.cwt(signal_data[1, :], scales, "gaus1", dt)
    power = (abs(cfs)) ** 2

    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    f, ax = plt.subplots()
    ax.contourf(time, np.log2(period), np.log2(power), np.log2(levels),
                extend='both')

    plt.figure()
    ax, fig = plt.subplots(2, 1)
    ax[0].plot(signal)
    ax[1].plot(signal_data[1, :])
    # Preprocessing OWN Try

    titles = ["db1", "db2", "db3", "db4", "gaus1", "gaus2", "gaus3", "gaus4"]

    fig, axs = plt.subplots(4, 2)

    for (ax1, ax2), i in zip(axs, range(0, 4)):
        cA, cD = pywt.dwt(signal_data[1, :], titles[int(i)])
        coef, freqs = pywt.cwt(signal_data[1, :], np.arange(1, 500), titles[i + 4])
        ax1.plot(cA)

        ax1.set_title(titles[i])

        ax2.matshow(coef)
        ax2.set_title(titles[i + 4])


'''''
def angle(directions):
    """
    Return the angle between vectors
    """
    vec2 = directions[1:]
    vec1 = directions[:-1]

    norm1 = np.sqrt((vec1 ** 2).sum(axis=1))
    norm2 = np.sqrt((vec2 ** 2).sum(axis=1))
    cos = (vec1 * vec2).sum(axis=1) / (norm1 * norm2)
    return np.arccos(cos)
'''''

t = np.arange(0, ECGconv.__len__())
wavlist = pywt.wavelist(kind='continuous')
fig, axs = plt.subplots(wavlist.__len__(), 1, sharex=True)
plt.plot(ECGconv)

for t in range(1, wavlist.__len__()):
    cwtmatr, freqs = pywt.cwt(ECGconv, np.arange(1, 512), wavelet=wavlist[t], sampling_period=1 / signal_f_s)
    axs[t].pcolormesh(t, freqs, cwtmatr, vmin=0, cmap="inferno")
    axs[t].title(wavlist[t])

"""""
            # ----------------------------------------------------------------------------------------------------------
            # TODO: Impl both possibilities??
            # Methode 1:
            # according to one study highest T-Amplitude (V1-V5)
            # Methode 2:
            # according to another in V5

            # Method 1
            t_max_amp = np.zeros(5)

            leads_1 = np.arange(7, 11, 1)
            counter_1 = 0
            t_peak_col = ["T-Peak-V2", "T-Peak-V3", "T-Peak-V4", "T-Peak-V5"]
            t_peak_corr = np.zeros((idx_df["T-Peak"].size, t_peak_col.__len__()), dtype=np.int16)

            for j in leads_1:

                # remove Baseline
                ECGblr = remove_baseline_fun(signal_data[j, :], signal_f_s)

                # Get every T-Ampl. individually --> different Peaks in Leads!!
                # +- 20 samples
                search_area = 30
                counter_2 = 0
                for jj in idx_df["T-Peak"]:

                    corr_t_idx = np.argmax(ECGblr[jj - search_area: jj + search_area])
                    t_peak_corr[counter_2, counter_1] = jj - search_area + corr_t_idx
                    counter_2 += 1

                counter_1 += 1
                #plt.plot(ECGblr)

            # get T-Ampl from Lead V2-V5
            t_peak_idx_df = pd.DataFrame(t_peak_corr, columns=t_peak_col, dtype=np.int16)

            # get features from Lead with highest T-Value
            t_max_amp = np.zeros(t_peak_col.__len__())
            counter = 0
            for lead, column in zip(leads_1, t_peak_idx_df):

                amp_data = t_ampl_fun(signal_data[lead, :], t_peak_idx_df, column)
                t_max_amp[counter] = amp_data[1]
                counter += 1

            idx_t_high = leads_1[np.argmax(t_max_amp)]
            idx_t_high_df = np.argmax(t_max_amp)
            # replace T-Peaks from Lead1 Dataframe
            idx_df["T-Peak"] = t_peak_idx_df[t_peak_col[idx_t_high_df]].values

            # Desired signal for feature extraction
            signal_T_high = signal_data[idx_t_high, :]

            # filter + baseline remove
            signal_T_high_br = remove_baseline_fun(signal_T_high, signal_f_s)

            # Box Filter
            signal_T_conv = convolve(signal_T_high_br, kernel=Box1DKernel(7.5))
            #-----------------------------------------------------------------------------------------------------------
            """""


def idx_finder_fun(ecg_object, filtered_signal, unfiltered_signal):
    """""    plt.figure(1)
    plt.scatter(features_scaled_std['JT_scaled'].loc[features.Sex == 0], features_scaled_std['T_desc_scaled'].loc[features.Sex == 0],
                marker='o')
    plt.scatter(features_scaled_std['JT_scaled'].loc[features.Sex == 1],
                features_scaled_std['T_desc_scaled'].loc[features.Sex == 1], marker='^')
    Find indices of ECG wave.
    R-Peak, J-Point, T-Peak, S-Peak
    Parameters:
    ----------
    ecg_object:         Tuple      ECG signal preprocessed
    filtered_signal:    np.array   filtered ECG signal
    unfiltered_signal:  np.array   unfiltered ECG signal
    Returns:
    ---------
    idx_dataframe:      pd.DataFrame DataFrame with indices
    """""
    idx_args = np.zeros((int(ecg_object['rpeaks'].size), 4))

    # Loop over R peaks to detect different POI Indexes (T-Peak, R-Peak, S-Peak, J-Point)
    interest_idx = 40
    idx_count = 0
    for ii in ecg_object['rpeaks']:

        # get area of interest
        area_of_interest = filtered_signal[ii - interest_idx: ii + interest_idx]
        area_of_interest_unf = unfiltered_signal[ii - interest_idx: ii + interest_idx]

        # gradient
        area_of_interest_dt = np.gradient(area_of_interest)

        # get zero crossings before
        zero_crossings_1 = np.where(np.gradient(np.sign(area_of_interest_dt)))[0]
        # TODO: delete R-Peak zero crossing > 1
        # get index of S == 1st deviation lowest point after R-Peak
        s_peak_idx = zero_crossings_1[np.argmax(zero_crossings_1 > 42)]

        # TODO: +1 ?! check with length?
        # distance R -> S [samples]
        # dist_R_S = s_peak_idx - 40
        # in msec (1ms == 5 samples)
        # dist_R_S_msec = dist_R_S * 0.002

        # TODO: crosscheck with second diff
        # TODO: crosscheck with 12std algorithm ; mean of 2nd diff of all 12 Leads
        # Search for J Point after S --> first smallest gradient
        # get highest value of diff
        data_after_S = area_of_interest[s_peak_idx:]
        data_after_S_unf = area_of_interest_unf[s_peak_idx:]
        s_gradient = np.gradient(data_after_S)

        # TODO: BUUUUUG!!!!!
        if s_gradient[5:].size == 0:
            breakpoint()

        idx_maxslope = np.argmax(s_gradient[5:])
        idx_minslope = np.argmin(s_gradient[5:])

        val_maxslope = np.max(s_gradient[5:])
        val_minslope = np.min(s_gradient[5:])

        # get diff and find first negative turning point
        slope_diff = np.diff(data_after_S_unf)
        idx_diff_tp = np.where(slope_diff < 0)


        # check time variable min 10 samples to J-Point
        # TODO: validate time var with beats/min?
        time_var = 13
        try:
            # TODO: BUG!!!!!!!!!!
            idx_diff_tp_time = idx_diff_tp[0][np.min(np.where(idx_diff_tp[0] >= time_var))]
            j_point_idx_2 = s_peak_idx + idx_diff_tp_time
        except ValueError:
            j_point_idx_2 = s_peak_idx + idx_minslope

        # check trend before and after +- 5 samples

        # get J Point level --> minus baseline (is it zero?????)

        J_point_level = 0 - area_of_interest[j_point_idx_2]


        # T-Peak Amplitude + 150 samples == 0.3sec
        # diff has to be zero at T-Peak
        # max point after J-Point
        area_of_interest_2 = unfiltered_signal[ii + (s_peak_idx - interest_idx): ii + 150]

        area_of_interest_dt_2 = np.gradient(area_of_interest_2)

        zero_crossings_2 = np.where(np.gradient(np.sign(area_of_interest_dt_2)))[0]

        t_peak_idx = np.argmax(area_of_interest_2)
        t_peak = np.max(area_of_interest_2)

        # TODO: Crosscheck with gradient

        #plt.figure(2)
        #plt.plot(area_of_interest_2)
        #plt.plot(t_peak_idx, t_peak, 'r*')

        # store idx
        idx_args[idx_count] = [ii, ii + s_peak_idx - interest_idx, ii + j_point_idx_2 - interest_idx,
                               ii + (s_peak_idx - interest_idx) + t_peak_idx]
        idx_count += 1

    idx_dataframe = pd.DataFrame(idx_args, columns=["R-Peak", "S-Peak", "J-Point", "T-Peak"],
                                 dtype=np.int16)
    return idx_dataframe



def train_svm(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    plot_roc_curve(classifier, X_test, y_test)

    print()
    print('Accuracy for training set svm = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set svm = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))


def train_naive_bayes(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.naive_bayes import GaussianNB

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print()
    print('Accuracy for training set Naive Bayes = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set Naive Bayes = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))


def train_log_reg(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print()
    print('Accuracy for training set Logistic Regression = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))


def train_decision_tree(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.tree import DecisionTreeClassifier

    classifier = DecisionTreeClassifier(max_depth=10)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print()
    print('Accuracy for training set Decision Tree = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set Decision Tree = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))


def train_random_forest(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier(n_estimators=5)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print()
    print('Accuracy for training set Random Forest = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test set Random Forest = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))


def train_mlp(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = mlp.predict(X_train)

    for i in range(0, len(y_pred_train)):
        if y_pred_train[i] >= 0.5:
            y_pred_train[i] = 1
        else:
            y_pred_train[i] = 0

    cm_train = confusion_matrix(y_pred_train, y_train)
    print()
    print('Accuracy for training MLP = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test MLP = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))


def train_adaboost(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.ensemble import AdaBoostClassifier

    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = ada.predict(X_train)

    for i in range(0, len(y_pred_train)):
        if y_pred_train[i] >= 0.5:
            y_pred_train[i] = 1
        else:
            y_pred_train[i] = 0

    cm_train = confusion_matrix(y_pred_train, y_train)
    print()
    print('Accuracy for training AdaBoost = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test AdaBoost = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))


def train_knn(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = knn.predict(X_train)

    for i in range(0, len(y_pred_train)):
        if y_pred_train[i] >= 0.5:  # setting threshold to .5
            y_pred_train[i] = 1
        else:
            y_pred_train[i] = 0

    cm_train = confusion_matrix(y_pred_train, y_train)


    print()
    print('Accuracy for training KNN = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test KNN = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))


def train_xgboost(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from xgboost import XGBClassifier

    xg = XGBClassifier()
    xg.fit(X_train, y_train)
    y_pred = xg.predict(X_test)

    from sklearn.metrics import confusion_matrix

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = xg.predict(X_train)

    for i in range(0, len(y_pred_train)):
        if y_pred_train[i] >= 0.5:
            y_pred_train[i] = 1
        else:
            y_pred_train[i] = 0

    cm_train = confusion_matrix(y_pred_train, y_train)
    print()
    print('Accuracy for training XGBoost = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test XGBoost = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))


def train_qda(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    y_pred = qda.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = qda.predict(X_train)

    for i in range(0, len(y_pred_train)):
        if y_pred_train[i] >= 0.5:
            y_pred_train[i] = 1
        else:
            y_pred_train[i] = 0

    cm_train = confusion_matrix(y_pred_train, y_train)
    print()
    print('Accuracy for training QDA = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
    print('Accuracy for test QDA = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))
