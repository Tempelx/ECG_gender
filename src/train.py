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
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt


def train_classifiers(X, y, comment, i):
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GridSearchCV

    classifiers = [SVC(kernel='rbf', probability=True),
                   SVC(kernel="linear", probability=True),
                   GaussianNB(),
                   LogisticRegression(),
                   DecisionTreeClassifier(),
                   RandomForestClassifier(),
                   MLPClassifier(),
                   AdaBoostClassifier(),
                   KNeighborsClassifier(),
                   XGBClassifier(),
                   QuadraticDiscriminantAnalysis()]

    names = ['RBF SVM', 'Linear SVM', 'Naive Bayes', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'MLP',
             'AdaBoost', 'KNN', 'XGBoost', 'QDA']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000, 10000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    """""
    plt.ioff()
    plt.figure(i)
    for name, clf in zip(names, classifiers):

        clf.fit(X_train, y_train)

        y_pred_test = clf.predict(X_test)
        cm_test = confusion_matrix(y_pred_test, y_test)

        y_pred_train = clf.predict(X_train)
        cm_train = confusion_matrix(y_pred_train, y_train)

        y_pred_proba_test = clf.predict_proba(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test[:, 1])
        auc = roc_auc_score(y_test, y_pred_proba_test[:, 1])
        plt.plot(fpr, tpr, label=name+", auc=" + str(round(auc, 3)))
        plt.legend(loc=4)

        print('Accuracy for ', name)
        print('Training set = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_pred_train)))
        print('Test set = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_pred_test)))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(comment)
    plt.ioff()
    """""

