#!/usr/bin/env python
__author__ = "Felix Tempel"
__copyright__ = "Copyright 2020, ECG Sex Classification"
__credits__ = ["Felix Tempel"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Felix Tempel"
__email__ = "felixtempel95@hotmail.de"
__status__ = "Production"

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd


class ParameterEstimator:

    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, **grid_kwargs):
        # GridSearch over all Classifiers
        for key in self.keys:
            print('Running GridSearchCV for %s.' % key)
            model = self.models[key]
            params = self.params[key]
            grid_search = GridSearchCV(model, params, **grid_kwargs)
            grid_search.fit(X, y)
            self.grid_searches[key] = grid_search
        print('Done.')

    def fit_test(self, X_test, y_test, X_train, y_train, df, comment):
        # fit first 5 best Classifiers and plot ROC curve
        plt.figure()
        for args in df.head(5).itertuples():
            model = self.models[args.estimator]
            model.set_params(**self.grid_searches.get(args.estimator).best_params_)

            model.fit(X_train, y_train)
            model.score(X_test, y_test)

            y_pred_test = model.predict(X_test)
            cm_test = confusion_matrix(y_pred_test, y_test)

            y_pred_train = model.predict(X_train)
            cm_train = confusion_matrix(y_pred_train, y_train)

            y_pred_proba_test = model.predict_proba(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test[:, 1])
            auc = roc_auc_score(y_test, y_pred_proba_test[:, 1])
            plt.plot(fpr, tpr, label=args.estimator + str(args.params) + ", auc=" + str(round(auc, 3)))
            plt.legend(loc=4)

            print('Accuracy for ', args.estimator)
            print('Training set = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_pred_train)))
            print('Test set = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_pred_test)))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title(comment)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    def score_summary(self, sort_by='mean_test_score'):
        frames = []
        for name, grid_search in self.grid_searches.items():
            frame = pd.DataFrame(grid_search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame) * [name]
            frames.append(frame)
        df = pd.concat(frames)

        df = df.sort_values([sort_by], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)

        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator'] + columns
        df = df[columns]
        return df


def train_classifiers(X, y, comment):

    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import AdaBoostClassifier

    models = {
        'SVC': SVC(probability=True),
        'GaussianNB': GaussianNB(),
        'LogisticRegression': LogisticRegression(),
        'MLPClassifier': MLPClassifier(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'XGBClassifier': XGBClassifier(),
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
        'ExtraTreesClassifier': ExtraTreesClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier()
    }

    tuned_parameters = {
        'SVC': {'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000, 10000]},
        'GaussianNB': {},
        'LogisticRegression': {'C': [0.1, 0.5, 1, 10, 100]},
        'MLPClassifier': [{'solver': ['lbfgs', 'sgd', 'adam']},
                          {'alpha': [0.0001, 0.00001, 0.0010]}],
        'KNeighborsClassifier': {'n_neighbors': [5, 10, 15, 20]},
        'XGBClassifier': {},
        'QuadraticDiscriminantAnalysis': {},
        'ExtraTreesClassifier': {'n_estimators': [16, 32]},
        'RandomForestClassifier': [
            {'n_estimators': [16, 32]},
            {'criterion': ['gini', 'entropy'],
             'n_estimators': [8, 16]}],
        'AdaBoostClassifier': {'n_estimators': [16, 32]},
        'GradientBoostingClassifier': {'n_estimators': [16, 32],
                                       'learning_rate': [0.8, 1.0]}
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    trainer = ParameterEstimator(models, tuned_parameters)

    trainer.fit(X_train, y_train, scoring='f1', n_jobs=2)

    trainer.fit_test(X_test, y_test, X_train, y_train, trainer.score_summary(), comment)

    return trainer.score_summary()


