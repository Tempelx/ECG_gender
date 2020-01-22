#!/usr/bin/env python
__author__ = "Felix Tempel"
__copyright__ = "Copyright 2020, ECG Sex Classification"
__credits__ = ["Felix Tempel"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Felix Tempel"
__email__ = "felixtempel95@hotmail.de"
__status__ = "Production"

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_roc_curve


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
