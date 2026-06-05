from copy import deepcopy

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y
import numpy as np


# Wrappers of calibration defined by Alexandari et al. in paper <http://proceedings.mlr.press/v119/alexandari20a.html>
# requires "pip install abstension"
# see https://github.com/kundajelab/abstention


def _require_abstention_calibration():
    try:
        from abstention.calibration import NoBiasVectorScaling, TempScaling, VectorScaling
    except ImportError as exc:
        raise ImportError(
            "Calibration methods in quapy.classification.calibration require the optional "
            "'abstention' package."
        ) from exc
    return NoBiasVectorScaling, TempScaling, VectorScaling


class RecalibratedProbabilisticClassifier:
    """
    Abstract class for (re)calibration method from `abstention.calibration`, as defined in
    `Alexandari, A., Kundaje, A., & Shrikumar, A. (2020, November). Maximum likelihood with bias-corrected calibration
    is hard-to-beat at label shift adaptation. In International Conference on Machine Learning (pp. 222-232). PMLR.
    <http://proceedings.mlr.press/v119/alexandari20a.html>`_:
    """
    pass


class RecalibratedProbabilisticClassifierBase(BaseEstimator, RecalibratedProbabilisticClassifier):
    """
    Applies a (re)calibration method from `abstention.calibration`, as defined in
    `Alexandari et al. paper <http://proceedings.mlr.press/v119/alexandari20a.html>`_.


    :param classifier: a scikit-learn probabilistic classifier
    :param calibrator: the calibration object (an instance of abstention.calibration.CalibratorFactory)
    :param val_split: indicate an integer k for performing kFCV to obtain the posterior probabilities, or a float p
        in (0,1) to indicate that the posteriors are obtained in a stratified validation split containing p% of the
        training instances (the rest is used for training). In any case, the classifier is retrained in the whole
        training set afterwards. Default value is 5.
    :param n_jobs: indicate the number of parallel workers (only when val_split is an integer); default=None
    :param verbose: whether or not to display information in the standard output
    """

    def __init__(self, classifier, calibrator, val_split=5, n_jobs=None, verbose=False):
        self.classifier = classifier
        self.calibrator = calibrator
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fits the calibration for the probabilistic classifier.

        :param X: array-like of shape `(n_samples, n_features)` with the data instances
        :param y: array-like of shape `(n_samples,)` with the class labels
        :return: self
        """
        k = self.val_split
        if isinstance(k, int):
            if k < 2:
                raise ValueError('wrong value for val_split: the number of folds must be > 2')
            return self.fit_cv(X, y)
        elif isinstance(k, float):
            if not (0 < k < 1):
                raise ValueError('wrong value for val_split: the proportion of validation documents must be in (0,1)')
            return self.fit_tr_val(X, y)

    def fit_cv(self, X, y):
        """
        Fits the calibration in a cross-validation manner, i.e., it generates posterior probabilities for all
        training instances via cross-validation, and then retrains the classifier on all training instances.
        The posterior probabilities thus generated are used for calibrating the outputs of the classifier.

        :param X: array-like of shape `(n_samples, n_features)` with the data instances
        :param y: array-like of shape `(n_samples,)` with the class labels
        :return: self
        """
        posteriors = cross_val_predict(
            self.classifier, X, y, cv=self.val_split, n_jobs=self.n_jobs, verbose=self.verbose, method='predict_proba'
        )
        self.classifier.fit(X, y)
        nclasses = len(np.unique(y))
        self.calibration_function = self.calibrator(posteriors, np.eye(nclasses)[y], posterior_supplied=True)
        return self

    def fit_tr_val(self, X, y):
        """
        Fits the calibration in a train/val-split manner, i.e.t, it partitions the training instances into a
        training and a validation set, and then uses the training samples to learn classifier which is then used
        to generate posterior probabilities for the held-out validation data. These posteriors are used to calibrate
        the classifier. The classifier is not retrained on the whole dataset.

        :param X: array-like of shape `(n_samples, n_features)` with the data instances
        :param y: array-like of shape `(n_samples,)` with the class labels
        :return: self
        """
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=self.val_split, stratify=y)
        self.classifier.fit(Xtr, ytr)
        posteriors = self.classifier.predict_proba(Xva)
        nclasses = len(np.unique(yva))
        self.calibration_function = self.calibrator(posteriors, np.eye(nclasses)[yva], posterior_supplied=True)
        return self

    def predict(self, X):
        """
        Predicts class labels for the data instances in `X`

        :param X: array-like of shape `(n_samples, n_features)` with the data instances
        :return: array-like of shape `(n_samples,)` with the class label predictions
        """
        return self.classifier.predict(X)

    def predict_proba(self, X):
        """
        Generates posterior probabilities for the data instances in `X`

        :param X: array-like of shape `(n_samples, n_features)` with the data instances
        :return: array-like of shape `(n_samples, n_classes)` with posterior probabilities
        """
        posteriors = self.classifier.predict_proba(X)
        return self.calibration_function(posteriors)

    @property
    def classes_(self):
        """
        Returns the classes on which the classifier has been trained on

        :return: array-like of shape `(n_classes)`
        """
        return self.classifier.classes_


class NBVSCalibration(RecalibratedProbabilisticClassifierBase):
    """
    Applies the No-Bias Vector Scaling (NBVS) calibration method from `abstention.calibration`, as defined in
    `Alexandari et al. paper <http://proceedings.mlr.press/v119/alexandari20a.html>`_:

    :param classifier: a scikit-learn probabilistic classifier
    :param val_split: indicate an integer k for performing kFCV to obtain the posterior prevalences, or a float p
        in (0,1) to indicate that the posteriors are obtained in a stratified validation split containing p% of the
        training instances (the rest is used for training). In any case, the classifier is retrained in the whole
        training set afterwards. Default value is 5.
    :param n_jobs: indicate the number of parallel workers (only when val_split is an integer)
    :param verbose: whether or not to display information in the standard output
    """

    def __init__(self, classifier, val_split=5, n_jobs=None, verbose=False):
        NoBiasVectorScaling, _, _ = _require_abstention_calibration()
        self.classifier = classifier
        self.calibrator = NoBiasVectorScaling(verbose=verbose)
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.verbose = verbose


class BCTSCalibration(RecalibratedProbabilisticClassifierBase):
    """
    Applies the Bias-Corrected Temperature Scaling (BCTS) calibration method from `abstention.calibration`, as defined in
    `Alexandari et al. paper <http://proceedings.mlr.press/v119/alexandari20a.html>`_:

    :param classifier: a scikit-learn probabilistic classifier
    :param val_split: indicate an integer k for performing kFCV to obtain the posterior prevalences, or a float p
        in (0,1) to indicate that the posteriors are obtained in a stratified validation split containing p% of the
        training instances (the rest is used for training). In any case, the classifier is retrained in the whole
        training set afterwards. Default value is 5.
    :param n_jobs: indicate the number of parallel workers (only when val_split is an integer)
    :param verbose: whether or not to display information in the standard output
    """

    def __init__(self, classifier, val_split=5, n_jobs=None, verbose=False):
        _, TempScaling, _ = _require_abstention_calibration()
        self.classifier = classifier
        self.calibrator = TempScaling(verbose=verbose, bias_positions='all')
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.verbose = verbose


class TSCalibration(RecalibratedProbabilisticClassifierBase):
    """
    Applies the Temperature Scaling (TS) calibration method from `abstention.calibration`, as defined in
    `Alexandari et al. paper <http://proceedings.mlr.press/v119/alexandari20a.html>`_:

    :param classifier: a scikit-learn probabilistic classifier
    :param val_split: indicate an integer k for performing kFCV to obtain the posterior prevalences, or a float p
        in (0,1) to indicate that the posteriors are obtained in a stratified validation split containing p% of the
        training instances (the rest is used for training). In any case, the classifier is retrained in the whole
        training set afterwards. Default value is 5.
    :param n_jobs: indicate the number of parallel workers (only when val_split is an integer)
    :param verbose: whether or not to display information in the standard output
    """

    def __init__(self, classifier, val_split=5, n_jobs=None, verbose=False):
        _, TempScaling, _ = _require_abstention_calibration()
        self.classifier = classifier
        self.calibrator = TempScaling(verbose=verbose)
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.verbose = verbose


class VSCalibration(RecalibratedProbabilisticClassifierBase):
    """
    Applies the Vector Scaling (VS) calibration method from `abstention.calibration`, as defined in
    `Alexandari et al. paper <http://proceedings.mlr.press/v119/alexandari20a.html>`_:

    :param classifier: a scikit-learn probabilistic classifier
    :param val_split: indicate an integer k for performing kFCV to obtain the posterior prevalences, or a float p
        in (0,1) to indicate that the posteriors are obtained in a stratified validation split containing p% of the
        training instances (the rest is used for training). In any case, the classifier is retrained in the whole
        training set afterwards. Default value is 5.
    :param n_jobs: indicate the number of parallel workers (only when val_split is an integer)
    :param verbose: whether or not to display information in the standard output
    """

    def __init__(self, classifier, val_split=5, n_jobs=None, verbose=False):
        _, _, VectorScaling = _require_abstention_calibration()
        self.classifier = classifier
        self.calibrator = VectorScaling(verbose=verbose)
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.verbose = verbose


class TemperatureScalingFromLogits(BaseEstimator):
    """
    Calibrates a matrix of logits by learning a temperature-scaling mapping
    with the calibration methods from `abstention.calibration`.

    This estimator is useful when the inputs are already logits produced by a
    pretrained classifier, and the goal is to transform them directly into
    calibrated posterior probabilities without retraining the underlying model.

    :param bias_corrected: if True, uses Bias-Corrected Temperature Scaling
        (BCTS); otherwise, uses standard Temperature Scaling (TS)
    :param verbose: whether the underlying calibrator should display progress
        information
    """

    def __init__(self, bias_corrected=False, verbose=False):
        self.bias_corrected = bias_corrected
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fits the logits calibrator.

        :param X: array-like of shape `(n_samples, n_classes)` containing
            logits
        :param y: array-like of shape `(n_samples,)` containing class labels
        :return: self
        """
        X, y = check_X_y(X, y)

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        n_classes = len(self.classes_)
        logits_dim = X.shape[1]
        if n_classes != logits_dim:
            raise ValueError(
                f'mismatch between the number of classes ({n_classes}) and the '
                f'dimensionality of the logits ({logits_dim})'
            )

        _, TempScaling, _ = _require_abstention_calibration()
        calibrator = TempScaling(
            verbose=self.verbose,
            bias_positions='all' if self.bias_corrected else [],
        )
        self.calibrator_ = calibrator
        self.calibration_function_ = calibrator(X, np.eye(n_classes)[y_enc])
        return self

    def predict_proba(self, X):
        """
        Converts logits into calibrated posterior probabilities.

        :param X: array-like of shape `(n_samples, n_classes)` containing
            logits
        :return: array-like of shape `(n_samples, n_classes)` with calibrated
            posterior probabilities
        """
        return self.calibration_function_(X)

    def predict(self, X):
        """
        Predicts class labels after calibration.

        :param X: array-like of shape `(n_samples, n_classes)` containing
            logits
        :return: array-like of shape `(n_samples,)` with class label
            predictions
        """
        posteriors = self.predict_proba(X)
        return self.label_encoder_.inverse_transform(np.argmax(posteriors, axis=1))
