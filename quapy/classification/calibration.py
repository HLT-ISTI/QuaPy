from copy import deepcopy

from abstention.calibration import NoBiasVectorScaling, TempScaling, VectorScaling
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_predict, train_test_split
import numpy as np


# Wrappers of calibration defined by Alexandari et al. in paper <http://proceedings.mlr.press/v119/alexandari20a.html>
# requires "pip install abstension"
# see https://github.com/kundajelab/abstention


class RecalibratedClassifier:
    pass


class RecalibratedClassifierBase(BaseEstimator, RecalibratedClassifier):
    """
    Applies a (re)calibration method from abstention.calibration, as defined in
    `Alexandari et al. paper <http://proceedings.mlr.press/v119/alexandari20a.html>`_:

    :param estimator: a scikit-learn probabilistic classifier
    :param calibrator: the calibration object (an instance of abstention.calibration.CalibratorFactory)
    :param val_split: indicate an integer k for performing kFCV to obtain the posterior prevalences, or a float p
        in (0,1) to indicate that the posteriors are obtained in a stratified validation split containing p% of the
        training instances (the rest is used for training). In any case, the classifier is retrained in the whole
        training set afterwards.
    :param n_jobs: indicate the number of parallel workers (only when val_split is an integer)
    :param verbose: whether or not to display information in the standard output
    """

    def __init__(self, estimator, calibrator, val_split=5, n_jobs=1, verbose=False):
        self.estimator = estimator
        self.calibrator = calibrator
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        k = self.val_split
        if isinstance(k, int):
            if k < 2:
                raise ValueError('wrong value for val_split: the number of folds must be > 2')
            return self.fit_cv(X, y)
        elif isinstance(k, float):
            if not (0 < k < 1):
                raise ValueError('wrong value for val_split: the proportion of validation documents must be in (0,1)')
            return self.fit_cv(X, y)

    def fit_cv(self, X, y):
        posteriors = cross_val_predict(
            self.estimator, X, y, cv=self.val_split, n_jobs=self.n_jobs, verbose=self.verbose, method="predict_proba"
        )
        self.estimator.fit(X, y)
        nclasses = len(np.unique(y))
        self.calibration_function = self.calibrator(posteriors, np.eye(nclasses)[y], posterior_supplied=True)
        return self

    def fit_tr_val(self, X, y):
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=self.val_split, stratify=y)
        self.estimator.fit(Xtr, ytr)
        posteriors = self.estimator.predict_proba(Xva)
        nclasses = len(np.unique(yva))
        self.calibrator = self.calibrator(posteriors, np.eye(nclasses)[yva], posterior_supplied=True)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        posteriors = self.estimator.predict_proba(X)
        return self.calibration_function(posteriors)

    @property
    def classes_(self):
        return self.estimator.classes_


class NBVSCalibration(RecalibratedClassifierBase):
    """
    Applies the No-Bias Vector Scaling (NBVS) calibration method from abstention.calibration, as defined in
    `Alexandari et al. paper <http://proceedings.mlr.press/v119/alexandari20a.html>`_:

    :param estimator: a scikit-learn probabilistic classifier
    :param val_split: indicate an integer k for performing kFCV to obtain the posterior prevalences, or a float p
        in (0,1) to indicate that the posteriors are obtained in a stratified validation split containing p% of the
        training instances (the rest is used for training). In any case, the classifier is retrained in the whole
        training set afterwards.
    :param n_jobs: indicate the number of parallel workers (only when val_split is an integer)
    :param verbose: whether or not to display information in the standard output
    """

    def __init__(self, estimator, val_split=5, n_jobs=1, verbose=False):
        self.estimator = estimator
        self.calibrator = NoBiasVectorScaling(verbose=verbose)
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.verbose = verbose


class BCTSCalibration(RecalibratedClassifierBase):
    """
    Applies the Bias-Corrected Temperature Scaling (BCTS) calibration method from abstention.calibration, as defined in
    `Alexandari et al. paper <http://proceedings.mlr.press/v119/alexandari20a.html>`_:

    :param estimator: a scikit-learn probabilistic classifier
    :param val_split: indicate an integer k for performing kFCV to obtain the posterior prevalences, or a float p
        in (0,1) to indicate that the posteriors are obtained in a stratified validation split containing p% of the
        training instances (the rest is used for training). In any case, the classifier is retrained in the whole
        training set afterwards.
    :param n_jobs: indicate the number of parallel workers (only when val_split is an integer)
    :param verbose: whether or not to display information in the standard output
    """

    def __init__(self, estimator, val_split=5, n_jobs=1, verbose=False):
        self.estimator = estimator
        self.calibrator = TempScaling(verbose=verbose, bias_positions='all')
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.verbose = verbose


class TSCalibration(RecalibratedClassifierBase):
    """
    Applies the Temperature Scaling (TS) calibration method from abstention.calibration, as defined in
    `Alexandari et al. paper <http://proceedings.mlr.press/v119/alexandari20a.html>`_:

    :param estimator: a scikit-learn probabilistic classifier
    :param val_split: indicate an integer k for performing kFCV to obtain the posterior prevalences, or a float p
        in (0,1) to indicate that the posteriors are obtained in a stratified validation split containing p% of the
        training instances (the rest is used for training). In any case, the classifier is retrained in the whole
        training set afterwards.
    :param n_jobs: indicate the number of parallel workers (only when val_split is an integer)
    :param verbose: whether or not to display information in the standard output
    """

    def __init__(self, estimator, val_split=5, n_jobs=1, verbose=False):
        self.estimator = estimator
        self.calibrator = TempScaling(verbose=verbose)
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.verbose = verbose


class VSCalibration(RecalibratedClassifierBase):
    """
    Applies the Vector Scaling (VS) calibration method from abstention.calibration, as defined in
    `Alexandari et al. paper <http://proceedings.mlr.press/v119/alexandari20a.html>`_:

    :param estimator: a scikit-learn probabilistic classifier
    :param val_split: indicate an integer k for performing kFCV to obtain the posterior prevalences, or a float p
        in (0,1) to indicate that the posteriors are obtained in a stratified validation split containing p% of the
        training instances (the rest is used for training). In any case, the classifier is retrained in the whole
        training set afterwards.
    :param n_jobs: indicate the number of parallel workers (only when val_split is an integer)
    :param verbose: whether or not to display information in the standard output
    """

    def __init__(self, estimator, val_split=5, n_jobs=1, verbose=False):
        self.estimator = estimator
        self.calibrator = VectorScaling(verbose=verbose)
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.verbose = verbose

