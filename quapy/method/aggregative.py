from abc import abstractmethod
from copy import deepcopy
from typing import Callable, Union
import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import quapy as qp
import quapy.functional as F
from quapy.classification.calibration import NBVSCalibration, BCTSCalibration, TSCalibration, VSCalibration
from quapy.classification.svmperf import SVMperf
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier, BinaryQuantifier, OneVsAllGeneric


# Abstract classes
# ------------------------------------

class AggregativeQuantifier(BaseQuantifier):
    """
    Abstract class for quantification methods that base their estimations on the aggregation of classification
    results. Aggregative Quantifiers thus implement a :meth:`classify` method and maintain a :attr:`classifier`
    attribute. Subclasses of this abstract class must implement the method :meth:`aggregate` which computes the
    aggregation of label predictions. The method :meth:`quantify` comes with a default implementation based on
    :meth:`classify` and :meth:`aggregate`.
    """

    @abstractmethod
    def fit(self, data: LabelledCollection, fit_classifier=True):
        """
        Trains the aggregative quantifier

        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
        :param fit_classifier: whether or not to train the learner (default is True). Set to False if the
            learner has been trained outside the quantifier.
        :return: self
        """
        ...

    @property
    def classifier(self):
        """
        Gives access to the classifier

        :return: the classifier (typically an sklearn's Estimator)
        """
        return self.classifier_

    @classifier.setter
    def classifier(self, classifier):
        """
        Setter for the classifier

        :param classifier: the classifier
        """
        self.classifier_ = classifier

    def classify(self, instances):
        """
        Provides the label predictions for the given instances. The predictions should respect the format expected by
        :meth:`aggregate`, i.e., posterior probabilities for probabilistic quantifiers, or crisp predictions for
        non-probabilistic quantifiers

        :param instances: array-like
        :return: np.ndarray of shape `(n_instances,)` with label predictions
        """
        return self.classifier.predict(instances)

    def quantify(self, instances):
        """
        Generate class prevalence estimates for the sample's instances by aggregating the label predictions generated
        by the classifier.

        :param instances: array-like
        :return: `np.ndarray` of shape `(n_classes)` with class prevalence estimates.
        """
        classif_predictions = self.classify(instances)
        return self.aggregate(classif_predictions)

    @abstractmethod
    def aggregate(self, classif_predictions: np.ndarray):
        """
        Implements the aggregation of label predictions.

        :param classif_predictions: `np.ndarray` of label predictions
        :return: `np.ndarray` of shape `(n_classes,)` with class prevalence estimates.
        """
        ...

    @property
    def classes_(self):
        """
        Class labels, in the same order in which class prevalence values are to be computed.
        This default implementation actually returns the class labels of the learner.

        :return: array-like
        """
        return self.classifier.classes_


class AggregativeProbabilisticQuantifier(AggregativeQuantifier):
    """
    Abstract class for quantification methods that base their estimations on the aggregation of posterior probabilities
    as returned by a probabilistic classifier. Aggregative Probabilistic Quantifiers thus extend Aggregative
    Quantifiers by implementing a _posterior_probabilities_ method returning values in [0,1] -- the posterior
    probabilities.
    """

    def classify(self, instances):
        return self.classifier.predict_proba(instances)


# Helper
# ------------------------------------
def _ensure_probabilistic(classifier):
    if not hasattr(classifier, 'predict_proba'):
        print(f'The learner {classifier.__class__.__name__} does not seem to be probabilistic. '
              f'The learner will be calibrated.')
        classifier = CalibratedClassifierCV(classifier, cv=5)
    return classifier


def _training_helper(classifier,
                     data: LabelledCollection,
                     fit_classifier: bool = True,
                     ensure_probabilistic=False,
                     val_split: Union[LabelledCollection, float] = None):
    """
    Training procedure common to all Aggregative Quantifiers.

    :param classifier: the learner to be fit
    :param data: the data on which to fit the learner. If requested, the data will be split before fitting the learner.
    :param fit_classifier: whether or not to fit the learner (if False, then bypasses any action)
    :param ensure_probabilistic: if True, guarantees that the resulting classifier implements predict_proba (if the
        learner is not probabilistic, then a CalibratedCV instance of it is trained)
    :param val_split: if specified as a float, indicates the proportion of training instances that will define the
        validation split (e.g., 0.3 for using 30% of the training set as validation data); if specified as a
        LabelledCollection, represents the validation split itself
    :return: the learner trained on the training set, and the unused data (a _LabelledCollection_ if train_val_split>0
        or None otherwise) to be used as a validation set for any subsequent parameter fitting
    """
    if fit_classifier:
        if ensure_probabilistic:
            classifier = _ensure_probabilistic(classifier)
        if val_split is not None:
            if isinstance(val_split, float):
                if not (0 < val_split < 1):
                    raise ValueError(f'train/val split {val_split} out of range, must be in (0,1)')
                train, unused = data.split_stratified(train_prop=1 - val_split)
            elif isinstance(val_split, LabelledCollection):
                train = data
                unused = val_split
            else:
                raise ValueError(
                    f'param "val_split" ({type(val_split)}) not understood; use either a float indicating the split '
                    'proportion, or a LabelledCollection indicating the validation split')
        else:
            train, unused = data, None

        if isinstance(classifier, BaseQuantifier):
            classifier.fit(train)
        else:
            classifier.fit(*train.Xy)
    else:
        if ensure_probabilistic:
            if not hasattr(classifier, 'predict_proba'):
                raise AssertionError('error: the learner cannot be calibrated since fit_classifier is set to False')
        unused = None
        if isinstance(val_split, LabelledCollection):
            unused = val_split

    return classifier, unused


def cross_generate_predictions(
        data,
        classifier,
        val_split,
        probabilistic,
        fit_classifier,
        n_jobs
):

    n_jobs = qp._get_njobs(n_jobs)

    if isinstance(val_split, int):
        assert fit_classifier == True, \
            'the parameters for the adjustment cannot be estimated with kFCV with fit_classifier=False'

        if probabilistic:
            classifier = _ensure_probabilistic(classifier)
            predict = 'predict_proba'
        else:
            predict = 'predict'
        y_pred = cross_val_predict(classifier, *data.Xy, cv=val_split, n_jobs=n_jobs, method=predict)
        class_count = data.counts()

        # fit the learner on all data
        classifier.fit(*data.Xy)
        y = data.y
        classes = data.classes_
    else:
        classifier, val_data = _training_helper(
            classifier, data, fit_classifier, ensure_probabilistic=probabilistic, val_split=val_split
        )
        y_pred = classifier.predict_proba(val_data.instances) if probabilistic else classifier.predict(val_data.instances)
        y = val_data.labels
        classes = val_data.classes_
        class_count = val_data.counts()

    return classifier, y, y_pred, classes, class_count


# Methods
# ------------------------------------
class CC(AggregativeQuantifier):
    """
    The most basic Quantification method. One that simply classifies all instances and counts how many have been
    attributed to each of the classes in order to compute class prevalence estimates.

    :param classifier: a sklearn's Estimator that generates a classifier
    """

    def __init__(self, classifier: BaseEstimator):
        self.classifier = classifier

    def fit(self, data: LabelledCollection, fit_classifier=True):
        """
        Trains the Classify & Count method unless `fit_classifier` is False, in which case, the classifier is assumed to
        be already fit and there is nothing else to do.

        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
        :param fit_classifier: if False, the classifier is assumed to be fit
        :return: self
        """
        self.classifier, _ = _training_helper(self.classifier, data, fit_classifier)
        return self

    def aggregate(self, classif_predictions: np.ndarray):
        """
        Computes class prevalence estimates by counting the prevalence of each of the predicted labels.

        :param classif_predictions: array-like with label predictions
        :return: `np.ndarray` of shape `(n_classes,)` with class prevalence estimates.
        """
        return F.prevalence_from_labels(classif_predictions, self.classes_)


class ACC(AggregativeQuantifier):
    """
    `Adjusted Classify & Count <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_,
    the "adjusted" variant of :class:`CC`, that corrects the predictions of CC
    according to the `misclassification rates`.

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, classifier: BaseEstimator, val_split=0.4, n_jobs=None):
        self.classifier = classifier
        self.val_split = val_split
        self.n_jobs = qp._get_njobs(n_jobs)

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split: Union[float, int, LabelledCollection] = None):
        """
        Trains a ACC quantifier.

        :param data: the training set
        :param fit_classifier: set to False to bypass the training (the learner is assumed to be already fit)
        :param val_split: either a float in (0,1) indicating the proportion of training instances to use for
            validation (e.g., 0.3 for using 30% of the training set as validation data), or a LabelledCollection
            indicating the validation set itself, or an int indicating the number `k` of folds to be used in `k`-fold
            cross validation to estimate the parameters
        :return: self
        """

        if val_split is None:
            val_split = self.val_split

        self.classifier, y, y_, classes, class_count = cross_generate_predictions(
            data, self.classifier, val_split, probabilistic=False, fit_classifier=fit_classifier, n_jobs=self.n_jobs
        )

        self.cc = CC(self.classifier)
        self.Pte_cond_estim_ = self.getPteCondEstim(self.classifier.classes_, y, y_)

        return self

    @classmethod
    def getPteCondEstim(cls, classes, y, y_):
        # estimate the matrix with entry (i,j) being the estimate of P(yi|yj), that is, the probability that a
        # document that belongs to yj ends up being classified as belonging to yi
        conf = confusion_matrix(y, y_, labels=classes).T
        conf = conf.astype(float)
        class_counts = conf.sum(axis=0)
        for i, _ in enumerate(classes):
            if class_counts[i] == 0:
                conf[i, i] = 1
            else:
                conf[:, i] /= class_counts[i]
        return conf

    def classify(self, data):
        return self.cc.classify(data)

    def aggregate(self, classif_predictions):
        prevs_estim = self.cc.aggregate(classif_predictions)
        return ACC.solve_adjustment(self.Pte_cond_estim_, prevs_estim)

    @classmethod
    def solve_adjustment(cls, PteCondEstim, prevs_estim):
        """
        Solves the system linear system :math:`Ax = B` with :math:`A` = `PteCondEstim` and :math:`B` = `prevs_estim`

        :param PteCondEstim: a `np.ndarray` of shape `(n_classes,n_classes,)` with entry `(i,j)` being the estimate
            of :math:`P(y_i|y_j)`, that is, the probability that an instance that belongs to :math:`y_j` ends up being
            classified as belonging to :math:`y_i`
        :param prevs_estim: a `np.ndarray` of shape `(n_classes,)` with the class prevalence estimates
        :return: an adjusted `np.ndarray` of shape `(n_classes,)` with the corrected class prevalence estimates
        """
        A = PteCondEstim
        B = prevs_estim
        try:
            adjusted_prevs = np.linalg.solve(A, B)
            adjusted_prevs = np.clip(adjusted_prevs, 0, 1)
            adjusted_prevs /= adjusted_prevs.sum()
        except np.linalg.LinAlgError:
            adjusted_prevs = prevs_estim  # no way to adjust them!
        return adjusted_prevs


class PCC(AggregativeProbabilisticQuantifier):
    """
    `Probabilistic Classify & Count <https://ieeexplore.ieee.org/abstract/document/5694031>`_,
    the probabilistic variant of CC that relies on the posterior probabilities returned by a probabilistic classifier.

    :param classifier: a sklearn's Estimator that generates a classifier
    """

    def __init__(self, classifier: BaseEstimator):
        self.classifier = classifier

    def fit(self, data: LabelledCollection, fit_classifier=True):
        self.classifier, _ = _training_helper(self.classifier, data, fit_classifier, ensure_probabilistic=True)
        return self

    def aggregate(self, classif_posteriors):
        return F.prevalence_from_probabilities(classif_posteriors, binarize=False)


class PACC(AggregativeProbabilisticQuantifier):
    """
    `Probabilistic Adjusted Classify & Count <https://ieeexplore.ieee.org/abstract/document/5694031>`_,
    the probabilistic variant of ACC that relies on the posterior probabilities returned by a probabilistic classifier.

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    :param n_jobs: number of parallel workers
    """

    def __init__(self, classifier: BaseEstimator, val_split=0.4, n_jobs=None):
        self.classifier = classifier
        self.val_split = val_split
        self.n_jobs = qp._get_njobs(n_jobs)

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split: Union[float, int, LabelledCollection] = None):
        """
        Trains a PACC quantifier.

        :param data: the training set
        :param fit_classifier: set to False to bypass the training (the learner is assumed to be already fit)
        :param val_split: either a float in (0,1) indicating the proportion of training instances to use for
         validation (e.g., 0.3 for using 30% of the training set as validation data), or a LabelledCollection
         indicating the validation set itself, or an int indicating the number k of folds to be used in kFCV
         to estimate the parameters
        :return: self
        """

        if val_split is None:
            val_split = self.val_split

        self.classifier, y, y_, classes, class_count = cross_generate_predictions(
            data, self.classifier, val_split, probabilistic=True, fit_classifier=fit_classifier, n_jobs=self.n_jobs
        )

        self.pcc = PCC(self.classifier)
        self.Pte_cond_estim_ = self.getPteCondEstim(classes, y, y_)

        return self

    @classmethod
    def getPteCondEstim(cls, classes, y, y_):
        # estimate the matrix with entry (i,j) being the estimate of P(yi|yj), that is, the probability that a
        # document that belongs to yj ends up being classified as belonging to yi
        n_classes = len(classes)
        confusion = np.eye(n_classes)
        for i, class_ in enumerate(classes):
            idx = y == class_
            if idx.any():
                confusion[i] = y_[idx].mean(axis=0)

        return confusion.T

    def aggregate(self, classif_posteriors):
        prevs_estim = self.pcc.aggregate(classif_posteriors)
        return ACC.solve_adjustment(self.Pte_cond_estim_, prevs_estim)

    def classify(self, data):
        return self.pcc.classify(data)


class EMQ(AggregativeProbabilisticQuantifier):
    """
    `Expectation Maximization for Quantification <https://ieeexplore.ieee.org/abstract/document/6789744>`_ (EMQ),
    aka `Saerens-Latinne-Decaestecker` (SLD) algorithm.
    EMQ consists of using the well-known `Expectation Maximization algorithm` to iteratively update the posterior
    probabilities generated by a probabilistic classifier and the class prevalence estimates obtained via
    maximum-likelihood estimation, in a mutually recursive way, until convergence.

    :param classifier: a sklearn's Estimator that generates a classifier
    :param exact_train_prev: set to True (default) for using, as the initial observation, the true training prevalence;
        or set to False for computing the training prevalence as an estimate, akin to PCC, i.e., as the expected
        value of the posterior probabilities of the training instances as suggested in
        `Alexandari et al. paper <http://proceedings.mlr.press/v119/alexandari20a.html>`_:
    :param recalib: a string indicating the method of recalibration. Available choices include "nbvs" (No-Bias Vector
        Scaling), "bcts" (Bias-Corrected Temperature Scaling), "ts" (Temperature Scaling), and "vs" (Vector Scaling).
        The default value is None, indicating no recalibration.
    """

    MAX_ITER = 1000
    EPSILON = 1e-4

    def __init__(self, classifier: BaseEstimator, exact_train_prev=True, recalib=None):
        self.classifier = classifier
        self.exact_train_prev = exact_train_prev
        self.recalib = recalib

    def fit(self, data: LabelledCollection, fit_classifier=True):
        if self.recalib is not None:
            if self.recalib == 'nbvs':
                self.classifier = NBVSCalibration(self.classifier)
            elif self.recalib == 'bcts':
                self.classifier = BCTSCalibration(self.classifier)
            elif self.recalib == 'ts':
                self.classifier = TSCalibration(self.classifier)
            elif self.recalib == 'vs':
                self.classifier = VSCalibration(self.classifier)
            elif self.recalib == 'platt':
                self.classifier = CalibratedClassifierCV(self.classifier, ensemble=False)
            else:
                raise ValueError('invalid param argument for recalibration method; available ones are '
                                 '"nbvs", "bcts", "ts", and "vs".')
        self.classifier, _ = _training_helper(self.classifier, data, fit_classifier, ensure_probabilistic=True)
        if self.exact_train_prev:
            self.train_prevalence = F.prevalence_from_labels(data.labels, self.classes_)
        else:
            self.train_prevalence = qp.model_selection.cross_val_predict(
                quantifier=PCC(deepcopy(self.classifier)),
                data=data,
                nfolds=3,
                random_state=0
            )
        return self

    def aggregate(self, classif_posteriors, epsilon=EPSILON):
        priors, posteriors = self.EM(self.train_prevalence, classif_posteriors, epsilon)
        return priors

    def predict_proba(self, instances, epsilon=EPSILON):
        classif_posteriors = self.classifier.predict_proba(instances)
        priors, posteriors = self.EM(self.train_prevalence, classif_posteriors, epsilon)
        return posteriors

    @classmethod
    def EM(cls, tr_prev, posterior_probabilities, epsilon=EPSILON):
        """
        Computes the `Expectation Maximization` routine.

        :param tr_prev: array-like, the training prevalence
        :param posterior_probabilities: `np.ndarray` of shape `(n_instances, n_classes,)` with the
            posterior probabilities
        :param epsilon: float, the threshold different between two consecutive iterations
            to reach before stopping the loop
        :return: a tuple with the estimated prevalence values (shape `(n_classes,)`) and
            the corrected posterior probabilities (shape `(n_instances, n_classes,)`)
        """
        Px = posterior_probabilities
        Ptr = np.copy(tr_prev)
        qs = np.copy(Ptr)  # qs (the running estimate) is initialized as the training prevalence

        s, converged = 0, False
        qs_prev_ = None
        while not converged and s < EMQ.MAX_ITER:
            # E-step: ps is Ps(y|xi)
            ps_unnormalized = (qs / Ptr) * Px
            ps = ps_unnormalized / ps_unnormalized.sum(axis=1, keepdims=True)

            # M-step:
            qs = ps.mean(axis=0)

            if qs_prev_ is not None and qp.error.mae(qs, qs_prev_) < epsilon and s > 10:
                converged = True

            qs_prev_ = qs
            s += 1

        if not converged:
            print('[warning] the method has reached the maximum number of iterations; it might have not converged')

        return qs, ps


class HDy(AggregativeProbabilisticQuantifier, BinaryQuantifier):
    """
    `Hellinger Distance y <https://www.sciencedirect.com/science/article/pii/S0020025512004069>`_ (HDy).
    HDy is a probabilistic method for training binary quantifiers, that models quantification as the problem of
    minimizing the divergence (in terms of the Hellinger Distance) between two cumulative distributions of posterior
    probabilities returned by the classifier. One of the distributions is generated from the unlabelled examples and
    the other is generated from a validation set. This latter distribution is defined as a mixture of the
    class-conditional distributions of the posterior probabilities returned for the positive and negative validation
    examples, respectively. The parameters of the mixture thus represent the estimates of the class prevalence values.

    :param classifier: a sklearn's Estimator that generates a binary classifier
    :param val_split: a float in range (0,1) indicating the proportion of data to be used as a stratified held-out
        validation distribution, or a :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, classifier: BaseEstimator, val_split=0.4):
        self.classifier = classifier
        self.val_split = val_split

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split: Union[float, LabelledCollection] = None):
        """
        Trains a HDy quantifier.

        :param data: the training set
        :param fit_classifier: set to False to bypass the training (the learner is assumed to be already fit)
        :param val_split: either a float in (0,1) indicating the proportion of training instances to use for
         validation (e.g., 0.3 for using 30% of the training set as validation data), or a
         :class:`quapy.data.base.LabelledCollection` indicating the validation set itself
        :return: self
        """
        if val_split is None:
            val_split = self.val_split

        self._check_binary(data, self.__class__.__name__)
        self.classifier, validation = _training_helper(
            self.classifier, data, fit_classifier, ensure_probabilistic=True, val_split=val_split)
        Px = self.classify(validation.instances)[:, 1]  # takes only the P(y=+1|x)
        self.Pxy1 = Px[validation.labels == self.classifier.classes_[1]]
        self.Pxy0 = Px[validation.labels == self.classifier.classes_[0]]
        # pre-compute the histogram for positive and negative examples
        self.bins = np.linspace(10, 110, 11, dtype=int)  # [10, 20, 30, ..., 100, 110]
        self.Pxy1_density = {bins: np.histogram(self.Pxy1, bins=bins, range=(0, 1), density=True)[0] for bins in
                             self.bins}
        self.Pxy0_density = {bins: np.histogram(self.Pxy0, bins=bins, range=(0, 1), density=True)[0] for bins in
                             self.bins}
        return self

    def aggregate(self, classif_posteriors):
        # "In this work, the number of bins b used in HDx and HDy was chosen from 10 to 110 in steps of 10,
        # and the final estimated a priori probability was taken as the median of these 11 estimates."
        # (González-Castro, et al., 2013).

        Px = classif_posteriors[:, 1]  # takes only the P(y=+1|x)

        prev_estimations = []
        # for bins in np.linspace(10, 110, 11, dtype=int):  #[10, 20, 30, ..., 100, 110]
        # Pxy0_density, _ = np.histogram(self.Pxy0, bins=bins, range=(0, 1), density=True)
        # Pxy1_density, _ = np.histogram(self.Pxy1, bins=bins, range=(0, 1), density=True)
        for bins in self.bins:
            Pxy0_density = self.Pxy0_density[bins]
            Pxy1_density = self.Pxy1_density[bins]

            Px_test, _ = np.histogram(Px, bins=bins, range=(0, 1), density=True)

            prev_selected, min_dist = None, None
            for prev in F.prevalence_linspace(n_prevalences=100, repeats=1, smooth_limits_epsilon=0.0):
                Px_train = prev * Pxy1_density + (1 - prev) * Pxy0_density
                hdy = F.HellingerDistance(Px_train, Px_test)
                if prev_selected is None or hdy < min_dist:
                    prev_selected, min_dist = prev, hdy
            prev_estimations.append(prev_selected)

        class1_prev = np.median(prev_estimations)
        return np.asarray([1 - class1_prev, class1_prev])


def _get_divergence(divergence: Union[str, Callable]):
    if isinstance(divergence, str):
        if divergence=='HD':
            return F.HellingerDistance
        elif divergence=='topsoe':
            return F.TopsoeDistance
        else:
            raise ValueError(f'unknown divergence {divergence}')
    elif callable(divergence):
        return divergence
    else:
        raise ValueError(f'argument "divergence" not understood; use a str or a callable function')


class DyS(AggregativeProbabilisticQuantifier, BinaryQuantifier):
    """
    `DyS framework <https://ojs.aaai.org/index.php/AAAI/article/view/4376>`_ (DyS).
    DyS is a generalization of HDy method, using a Ternary Search in order to find the prevalence that
    minimizes the distance between distributions.
    Details for the ternary search have been got from <https://dl.acm.org/doi/pdf/10.1145/3219819.3220059>

    :param classifier: a sklearn's Estimator that generates a binary classifier
    :param val_split: a float in range (0,1) indicating the proportion of data to be used as a stratified held-out
        validation distribution, or a :class:`quapy.data.base.LabelledCollection` (the split itself).
    :param n_bins: an int with the number of bins to use to compute the histograms.
    :param divergence: a str indicating the name of divergence (currently supported ones are "HD" or "topsoe"), or a
        callable function computes the divergence between two distributions (two equally sized arrays).
    :param tol: a float with the tolerance for the ternary search algorithm.
    """

    def __init__(self, classifier: BaseEstimator, val_split=0.4, n_bins=8, divergence: Union[str, Callable]= 'HD', tol=1e-05):
        self.classifier = classifier
        self.val_split = val_split
        self.tol = tol
        self.divergence = divergence
        self.n_bins = n_bins

    def _ternary_search(self, f, left, right, tol):
        """
        Find maximum of unimodal function f() within [left, right]
        """
        while abs(right - left) >= tol:
            left_third = left + (right - left) / 3
            right_third = right - (right - left) / 3

            if f(left_third) > f(right_third):
                left = left_third
            else:
                right = right_third

        # Left and right are the current bounds; the maximum is between them
        return (left + right) / 2

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split: Union[float, LabelledCollection] = None):
        if val_split is None:
            val_split = self.val_split

        self._check_binary(data, self.__class__.__name__)
        self.classifier, validation = _training_helper(
            self.classifier, data, fit_classifier, ensure_probabilistic=True, val_split=val_split)
        Px = self.classify(validation.instances)[:, 1]  # takes only the P(y=+1|x)
        self.Pxy1 = Px[validation.labels == self.classifier.classes_[1]]
        self.Pxy0 = Px[validation.labels == self.classifier.classes_[0]]
        self.Pxy1_density = np.histogram(self.Pxy1, bins=self.n_bins, range=(0, 1), density=True)[0]
        self.Pxy0_density = np.histogram(self.Pxy0, bins=self.n_bins, range=(0, 1), density=True)[0]
        return self

    def aggregate(self, classif_posteriors):
        Px = classif_posteriors[:, 1]  # takes only the P(y=+1|x)

        Px_test = np.histogram(Px, bins=self.n_bins, range=(0, 1), density=True)[0]
        divergence = _get_divergence(self.divergence)

        def distribution_distance(prev):
            Px_train = prev * self.Pxy1_density + (1 - prev) * self.Pxy0_density
            return divergence(Px_train, Px_test)
            
        class1_prev = self._ternary_search(f=distribution_distance, left=0, right=1, tol=self.tol)
        return np.asarray([1 - class1_prev, class1_prev])


class SMM(AggregativeProbabilisticQuantifier, BinaryQuantifier):
    """
    `SMM method <https://ieeexplore.ieee.org/document/9260028>`_ (SMM).
    SMM is a simplification of matching distribution methods where the representation of the examples
    is created using the mean instead of a histogram.

    :param classifier: a sklearn's Estimator that generates a binary classifier.
    :param val_split: a float in range (0,1) indicating the proportion of data to be used as a stratified held-out
        validation distribution, or a :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, classifier: BaseEstimator, val_split=0.4):
        self.classifier = classifier
        self.val_split = val_split
      
    def fit(self, data: LabelledCollection, fit_classifier=True, val_split: Union[float, LabelledCollection] = None):
        if val_split is None:
            val_split = self.val_split

        self._check_binary(data, self.__class__.__name__)
        self.classifier, validation = _training_helper(
            self.classifier, data, fit_classifier, ensure_probabilistic=True, val_split=val_split)
        Px = self.classify(validation.instances)[:, 1]  # takes only the P(y=+1|x)
        self.Pxy1 = Px[validation.labels == self.classifier.classes_[1]]
        self.Pxy0 = Px[validation.labels == self.classifier.classes_[0]]
        self.Pxy1_mean = np.mean(self.Pxy1)
        self.Pxy0_mean = np.mean(self.Pxy0)
        return self

    def aggregate(self, classif_posteriors):
        Px = classif_posteriors[:, 1]  # takes only the P(y=+1|x)
        Px_mean = np.mean(Px)
     
        class1_prev = (Px_mean - self.Pxy0_mean)/(self.Pxy1_mean - self.Pxy0_mean)
        class1_prev = np.clip(class1_prev, 0, 1)

        return np.asarray([1 - class1_prev, class1_prev])


class DistributionMatching(AggregativeProbabilisticQuantifier):
    """
    Generic Distribution Matching quantifier for binary or multiclass quantification.
    This implementation takes the number of bins, the divergence, and the possibility to work on CDF as hyperparameters.

    :param classifier: a `sklearn`'s Estimator that generates a probabilistic classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set to model the
        validation distribution.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the validation distribution should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    :param nbins: number of bins used to discretize the distributions (default 8)
    :param divergence: a string representing a divergence measure (currently, "HD" and "topsoe" are implemented)
        or a callable function taking two ndarrays of the same dimension as input (default "HD", meaning Hellinger
        Distance)
    :param cdf: whether or not to use CDF instead of PDF (default False)
    :param n_jobs: number of parallel workers (default None)
    """

    def __init__(self, classifier, val_split=0.4, nbins=8, divergence: Union[str, Callable]='HD', cdf=False, n_jobs=None):
        self.classifier = classifier
        self.val_split = val_split
        self.nbins = nbins
        self.divergence = divergence
        self.cdf = cdf
        self.n_jobs = n_jobs

    def __get_distributions(self, posteriors):
        histograms = []
        post_dims = posteriors.shape[1]
        if post_dims == 2:
            # in binary quantification we can use only one class, since the other one is its complement
            post_dims = 1
        for dim in range(post_dims):
            hist = np.histogram(posteriors[:, dim], bins=self.nbins, range=(0, 1))[0]
            histograms.append(hist)

        counts = np.vstack(histograms)
        distributions = counts/counts.sum(axis=1)[:,np.newaxis]
        if self.cdf:
            distributions = np.cumsum(distributions, axis=1)
        return distributions

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split: Union[float, LabelledCollection] = None):
        """
        Trains the classifier (if requested) and generates the validation distributions out of the training data.
        The validation distributions have shape `(n, ch, nbins)`, with `n` the number of classes, `ch` the number of
        channels, and `nbins` the number of bins. In particular, let `V` be the validation distributions; `di=V[i]`
        are the distributions obtained from training data labelled with class `i`; `dij = di[j]` is the discrete
        distribution of posterior probabilities `P(Y=j|X=x)` for training data labelled with class `i`, and `dij[k]`
        is the fraction of instances with a value in the `k`-th bin.

        :param data: the training set
        :param fit_classifier: set to False to bypass the training (the learner is assumed to be already fit)
        :param val_split: either a float in (0,1) indicating the proportion of training instances to use for
         validation (e.g., 0.3 for using 30% of the training set as validation data), or a LabelledCollection
         indicating the validation set itself, or an int indicating the number k of folds to be used in kFCV
         to estimate the parameters
        """
        if val_split is None:
            val_split = self.val_split

        self.classifier, y, posteriors, classes, class_count = cross_generate_predictions(
            data, self.classifier, val_split, probabilistic=True, fit_classifier=fit_classifier, n_jobs=self.n_jobs
        )

        self.validation_distribution = np.asarray(
            [self.__get_distributions(posteriors[y==cat]) for cat in range(data.n_classes)]
        )

        return self

    def aggregate(self, posteriors: np.ndarray):
        """
        Searches for the mixture model parameter (the sought prevalence values) that yields a validation distribution
        (the mixture) that best matches the test distribution, in terms of the divergence measure of choice.
        In the multiclass case, with `n` the number of classes, the test and mixture distributions contain
        `n` channels (proper distributions of binned posterior probabilities), on which the divergence is computed
        independently. The matching is computed as an average of the divergence across all channels.

        :param instances: instances in the sample
        :return: a vector of class prevalence estimates
        """
        test_distribution = self.__get_distributions(posteriors)
        divergence = _get_divergence(self.divergence)
        n_classes, n_channels, nbins = self.validation_distribution.shape
        def match(prev):
            prev = np.expand_dims(prev, axis=0)
            mixture_distribution = (prev @ self.validation_distribution.reshape(n_classes,-1)).reshape(n_channels, -1)
            divs = [divergence(test_distribution[ch], mixture_distribution[ch]) for ch in range(n_channels)]
            return np.mean(divs)

        # the initial point is set as the uniform distribution
        uniform_distribution = np.full(fill_value=1 / n_classes, shape=(n_classes,))

        # solutions are bounded to those contained in the unit-simplex
        bounds = tuple((0, 1) for x in range(n_classes))  # values in [0,1]
        constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})  # values summing up to 1
        r = optimize.minimize(match, x0=uniform_distribution, method='SLSQP', bounds=bounds, constraints=constraints)
        return r.x


def newELM(svmperf_base=None, loss='01', C=1):
    """
    Explicit Loss Minimization (ELM) quantifiers.
    Quantifiers based on ELM represent a family of methods based on structured output learning;
    these quantifiers rely on classifiers that have been optimized using a quantification-oriented loss
    measure. This implementation relies on
    `Joachims’ SVM perf <https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html>`_ structured output
    learning algorithm, which has to be installed and patched for the purpose (see this
    `script <https://github.com/HLT-ISTI/QuaPy/blob/master/prepare_svmperf.sh>`_).
    This function equivalent to:

    >>> CC(SVMperf(svmperf_base, loss, C))

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`; if set to None (default)
        this path will be obtained from qp.environ['SVMPERF_HOME']
    :param loss: the loss to optimize (see :attr:`quapy.classification.svmperf.SVMperf.valid_losses`)
    :param C: trade-off between training error and margin (default 0.01)
    :return: returns an instance of CC set to work with SVMperf (with loss and C set properly) as the
        underlying classifier
    """
    if svmperf_base is None:
        svmperf_base = qp.environ['SVMPERF_HOME']
    assert svmperf_base is not None, \
        'param svmperf_base was not specified, and the variable SVMPERF_HOME has not been set in the environment'
    return CC(SVMperf(svmperf_base, loss=loss, C=C))


def newSVMQ(svmperf_base=None, C=1):
    """
    SVM(Q) is an Explicit Loss Minimization (ELM) quantifier set to optimize for the `Q` loss combining a
    classification-oriented loss and a quantification-oriented loss, as proposed by
    `Barranquero et al. 2015 <https://www.sciencedirect.com/science/article/pii/S003132031400291X>`_.
    Equivalent to:

    >>> CC(SVMperf(svmperf_base, loss='q', C=C))

    Quantifiers based on ELM represent a family of methods based on structured output learning;
    these quantifiers rely on classifiers that have been optimized using a quantification-oriented loss
    measure. This implementation relies on
    `Joachims’ SVM perf <https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html>`_ structured output
    learning algorithm, which has to be installed and patched for the purpose (see this
    `script <https://github.com/HLT-ISTI/QuaPy/blob/master/prepare_svmperf.sh>`_).
    This function is a wrapper around CC(SVMperf(svmperf_base, loss, C))

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`; if set to None (default)
        this path will be obtained from qp.environ['SVMPERF_HOME']
    :param C: trade-off between training error and margin (default 0.01)
    :return: returns an instance of CC set to work with SVMperf (with loss and C set properly) as the
        underlying classifier
    """
    return newELM(svmperf_base, loss='q', C=C)

def newSVMKLD(svmperf_base=None, C=1):
    """
    SVM(KLD) is an Explicit Loss Minimization (ELM) quantifier set to optimize for the Kullback-Leibler Divergence
    as proposed by `Esuli et al. 2015 <https://dl.acm.org/doi/abs/10.1145/2700406>`_.
    Equivalent to:

    >>> CC(SVMperf(svmperf_base, loss='kld', C=C))

    Quantifiers based on ELM represent a family of methods based on structured output learning;
    these quantifiers rely on classifiers that have been optimized using a quantification-oriented loss
    measure. This implementation relies on
    `Joachims’ SVM perf <https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html>`_ structured output
    learning algorithm, which has to be installed and patched for the purpose (see this
    `script <https://github.com/HLT-ISTI/QuaPy/blob/master/prepare_svmperf.sh>`_).
    This function is a wrapper around CC(SVMperf(svmperf_base, loss, C))

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`; if set to None (default)
        this path will be obtained from qp.environ['SVMPERF_HOME']
    :param C: trade-off between training error and margin (default 0.01)
    :return: returns an instance of CC set to work with SVMperf (with loss and C set properly) as the
        underlying classifier
    """
    return newELM(svmperf_base, loss='kld', C=C)


def newSVMKLD(svmperf_base=None, C=1):
    """
    SVM(KLD) is an Explicit Loss Minimization (ELM) quantifier set to optimize for the Kullback-Leibler Divergence
    normalized via the logistic function, as proposed by
    `Esuli et al. 2015 <https://dl.acm.org/doi/abs/10.1145/2700406>`_.
    Equivalent to:

    >>> CC(SVMperf(svmperf_base, loss='nkld', C=C))

    Quantifiers based on ELM represent a family of methods based on structured output learning;
    these quantifiers rely on classifiers that have been optimized using a quantification-oriented loss
    measure. This implementation relies on
    `Joachims’ SVM perf <https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html>`_ structured output
    learning algorithm, which has to be installed and patched for the purpose (see this
    `script <https://github.com/HLT-ISTI/QuaPy/blob/master/prepare_svmperf.sh>`_).
    This function is a wrapper around CC(SVMperf(svmperf_base, loss, C))

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`; if set to None (default)
        this path will be obtained from qp.environ['SVMPERF_HOME']
    :param C: trade-off between training error and margin (default 0.01)
    :return: returns an instance of CC set to work with SVMperf (with loss and C set properly) as the
        underlying classifier
    """
    return newELM(svmperf_base, loss='nkld', C=C)

def newSVMAE(svmperf_base=None, C=1):
    """
    SVM(KLD) is an Explicit Loss Minimization (ELM) quantifier set to optimize for the Absolute Error as first used by
    `Moreo and Sebastiani, 2021 <https://arxiv.org/abs/2011.02552>`_.
    Equivalent to:

    >>> CC(SVMperf(svmperf_base, loss='mae', C=C))

    Quantifiers based on ELM represent a family of methods based on structured output learning;
    these quantifiers rely on classifiers that have been optimized using a quantification-oriented loss
    measure. This implementation relies on
    `Joachims’ SVM perf <https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html>`_ structured output
    learning algorithm, which has to be installed and patched for the purpose (see this
    `script <https://github.com/HLT-ISTI/QuaPy/blob/master/prepare_svmperf.sh>`_).
    This function is a wrapper around CC(SVMperf(svmperf_base, loss, C))

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`; if set to None (default)
        this path will be obtained from qp.environ['SVMPERF_HOME']
    :param C: trade-off between training error and margin (default 0.01)
    :return: returns an instance of CC set to work with SVMperf (with loss and C set properly) as the
        underlying classifier
    """
    return newELM(svmperf_base, loss='mae', C=C)

def newSVMRAE(svmperf_base=None, C=1):
    """
    SVM(KLD) is an Explicit Loss Minimization (ELM) quantifier set to optimize for the Relative Absolute Error as first
    used by `Moreo and Sebastiani, 2021 <https://arxiv.org/abs/2011.02552>`_.
    Equivalent to:

    >>> CC(SVMperf(svmperf_base, loss='mrae', C=C))

    Quantifiers based on ELM represent a family of methods based on structured output learning;
    these quantifiers rely on classifiers that have been optimized using a quantification-oriented loss
    measure. This implementation relies on
    `Joachims’ SVM perf <https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html>`_ structured output
    learning algorithm, which has to be installed and patched for the purpose (see this
    `script <https://github.com/HLT-ISTI/QuaPy/blob/master/prepare_svmperf.sh>`_).
    This function is a wrapper around CC(SVMperf(svmperf_base, loss, C))

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`; if set to None (default)
        this path will be obtained from qp.environ['SVMPERF_HOME']
    :param C: trade-off between training error and margin (default 0.01)
    :return: returns an instance of CC set to work with SVMperf (with loss and C set properly) as the
        underlying classifier
    """
    return newELM(svmperf_base, loss='mrae', C=C)


class ThresholdOptimization(AggregativeQuantifier, BinaryQuantifier):
    """
    Abstract class of Threshold Optimization variants for :class:`ACC` as proposed by
    `Forman 2006 <https://dl.acm.org/doi/abs/10.1145/1150402.1150423>`_ and
    `Forman 2008 <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_.
    The goal is to bring improved stability to the denominator of the adjustment.
    The different variants are based on different heuristics for choosing a decision threshold
    that would allow for more true positives and many more false positives, on the grounds this
    would deliver larger denominators.

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, classifier: BaseEstimator, val_split=0.4, n_jobs=None):
        self.classifier = classifier
        self.val_split = val_split
        self.n_jobs = qp._get_njobs(n_jobs)

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split: Union[float, int, LabelledCollection] = None):
        self._check_binary(data, "Threshold Optimization")

        if val_split is None:
            val_split = self.val_split

        self.classifier, y, y_, classes, class_count = cross_generate_predictions(
            data, self.classifier, val_split, probabilistic=True, fit_classifier=fit_classifier, n_jobs=self.n_jobs
        )

        self.cc = CC(self.classifier)

        self.tpr, self.fpr = self._optimize_threshold(y, y_)

        return self

    @abstractmethod
    def _condition(self, tpr, fpr) -> float:
        """
        Implements the criterion according to which the threshold should be selected.
        This function should return the (float) score to be minimized.

        :param tpr: float, true positive rate
        :param fpr: float, false positive rate
        :return: float, a score for the given `tpr` and `fpr`
        """
        ...

    def _optimize_threshold(self, y, probabilities):
        """
        Seeks for the best `tpr` and `fpr` according to the score obtained at different
        decision thresholds. The scoring function is implemented in function `_condition`.

        :param y: predicted labels for the validation set (or for the training set via `k`-fold cross validation)
        :param probabilities: array-like with the posterior probabilities
        :return: best `tpr` and `fpr` according to `_condition`
        """
        best_candidate_threshold_score = None
        best_tpr = 0
        best_fpr = 0
        candidate_thresholds = np.unique(probabilities[:, 1])
        for candidate_threshold in candidate_thresholds:
            y_ = [self.classes_[1] if p > candidate_threshold else self.classes_[0] for p in probabilities[:, 1]]
            TP, FP, FN, TN = self._compute_table(y, y_)
            tpr = self._compute_tpr(TP, FP)
            fpr = self._compute_fpr(FP, TN)
            condition_score = self._condition(tpr, fpr)
            if best_candidate_threshold_score is None or condition_score < best_candidate_threshold_score:
                best_candidate_threshold_score = condition_score
                best_tpr = tpr
                best_fpr = fpr

        return best_tpr, best_fpr

    def aggregate(self, classif_predictions):
        prevs_estim = self.cc.aggregate(classif_predictions)
        if self.tpr - self.fpr == 0:
            return prevs_estim
        adjusted_prevs_estim = np.clip((prevs_estim[1] - self.fpr) / (self.tpr - self.fpr), 0, 1)
        adjusted_prevs_estim = np.array((1 - adjusted_prevs_estim, adjusted_prevs_estim))
        return adjusted_prevs_estim

    def _compute_table(self, y, y_):
        TP = np.logical_and(y == y_, y == self.classes_[1]).sum()
        FP = np.logical_and(y != y_, y == self.classes_[0]).sum()
        FN = np.logical_and(y != y_, y == self.classes_[1]).sum()
        TN = np.logical_and(y == y_, y == self.classes_[0]).sum()
        return TP, FP, FN, TN

    def _compute_tpr(self, TP, FP):
        if TP + FP == 0:
            return 1
        return TP / (TP + FP)

    def _compute_fpr(self, FP, TN):
        if FP + TN == 0:
            return 0
        return FP / (FP + TN)


class T50(ThresholdOptimization):
    """
    Threshold Optimization variant for :class:`ACC` as proposed by
    `Forman 2006 <https://dl.acm.org/doi/abs/10.1145/1150402.1150423>`_ and
    `Forman 2008 <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_ that looks
    for the threshold that makes `tpr` cosest to 0.5.
    The goal is to bring improved stability to the denominator of the adjustment.

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, classifier: BaseEstimator, val_split=0.4):
        super().__init__(classifier, val_split)

    def _condition(self, tpr, fpr) -> float:
        return abs(tpr - 0.5)


class MAX(ThresholdOptimization):
    """
    Threshold Optimization variant for :class:`ACC` as proposed by
    `Forman 2006 <https://dl.acm.org/doi/abs/10.1145/1150402.1150423>`_ and
    `Forman 2008 <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_ that looks
    for the threshold that maximizes `tpr-fpr`.
    The goal is to bring improved stability to the denominator of the adjustment.

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, classifier: BaseEstimator, val_split=0.4):
        super().__init__(classifier, val_split)

    def _condition(self, tpr, fpr) -> float:
        # MAX strives to maximize (tpr - fpr), which is equivalent to minimize (fpr - tpr)
        return (fpr - tpr)


class X(ThresholdOptimization):
    """
    Threshold Optimization variant for :class:`ACC` as proposed by
    `Forman 2006 <https://dl.acm.org/doi/abs/10.1145/1150402.1150423>`_ and
    `Forman 2008 <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_ that looks
    for the threshold that yields `tpr=1-fpr`.
    The goal is to bring improved stability to the denominator of the adjustment.

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, classifier: BaseEstimator, val_split=0.4):
        super().__init__(classifier, val_split)

    def _condition(self, tpr, fpr) -> float:
        return abs(1 - (tpr + fpr))


class MS(ThresholdOptimization):
    """
    Median Sweep. Threshold Optimization variant for :class:`ACC` as proposed by
    `Forman 2006 <https://dl.acm.org/doi/abs/10.1145/1150402.1150423>`_ and
    `Forman 2008 <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_ that generates
    class prevalence estimates for all decision thresholds and returns the median of them all.
    The goal is to bring improved stability to the denominator of the adjustment.

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """
    def __init__(self, classifier: BaseEstimator, val_split=0.4):
        super().__init__(classifier, val_split)

    def _condition(self, tpr, fpr) -> float:
        pass

    def _optimize_threshold(self, y, probabilities):
        tprs = []
        fprs = []
        candidate_thresholds = np.unique(probabilities[:, 1])
        for candidate_threshold in candidate_thresholds:
            y_ = [self.classes_[1] if p > candidate_threshold else self.classes_[0] for p in probabilities[:, 1]]
            TP, FP, FN, TN = self._compute_table(y, y_)
            tpr = self._compute_tpr(TP, FP)
            fpr = self._compute_fpr(FP, TN)
            tprs.append(tpr)
            fprs.append(fpr)
        return np.median(tprs), np.median(fprs)


class MS2(MS):
    """
    Median Sweep 2. Threshold Optimization variant for :class:`ACC` as proposed by
    `Forman 2006 <https://dl.acm.org/doi/abs/10.1145/1150402.1150423>`_ and
    `Forman 2008 <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_ that generates
    class prevalence estimates for all decision thresholds and returns the median of for cases in
    which `tpr-fpr>0.25`
    The goal is to bring improved stability to the denominator of the adjustment.

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """
    def __init__(self, classifier: BaseEstimator, val_split=0.4):
        super().__init__(classifier, val_split)

    def _optimize_threshold(self, y, probabilities):
        tprs = [0, 1]
        fprs = [0, 1]
        candidate_thresholds = np.unique(probabilities[:, 1])
        for candidate_threshold in candidate_thresholds:
            y_ = [self.classes_[1] if p > candidate_threshold else self.classes_[0] for p in probabilities[:, 1]]
            TP, FP, FN, TN = self._compute_table(y, y_)
            tpr = self._compute_tpr(TP, FP)
            fpr = self._compute_fpr(FP, TN)
            if (tpr - fpr) > 0.25:
                tprs.append(tpr)
                fprs.append(fpr)
        return np.median(tprs), np.median(fprs)


ClassifyAndCount = CC
AdjustedClassifyAndCount = ACC
ProbabilisticClassifyAndCount = PCC
ProbabilisticAdjustedClassifyAndCount = PACC
ExpectationMaximizationQuantifier = EMQ
SLD = EMQ
HellingerDistanceY = HDy
MedianSweep = MS
MedianSweep2 = MS2


class OneVsAllAggregative(OneVsAllGeneric, AggregativeQuantifier):
    """
    Allows any binary quantifier to perform quantification on single-label datasets.
    The method maintains one binary quantifier for each class, and then l1-normalizes the outputs so that the
    class prevelences sum up to 1.
    This variant was used, along with the :class:`EMQ` quantifier, in
    `Gao and Sebastiani, 2016 <https://link.springer.com/content/pdf/10.1007/s13278-016-0327-z.pdf>`_.

    :param binary_quantifier: a quantifier (binary) that will be employed to work on multiclass model in a
        one-vs-all manner
    :param n_jobs: number of parallel workers
    :param parallel_backend: the parallel backend for joblib (default "loky"); this is helpful for some quantifiers
        (e.g., ELM-based ones) that cannot be run with multiprocessing, since the temp dir they create during fit will
        is removed and no longer available at predict time.
    """

    def __init__(self, binary_quantifier, n_jobs=None, parallel_backend='multiprocessing'):
        assert isinstance(binary_quantifier, BaseQuantifier), \
            f'{self.binary_quantifier} does not seem to be a Quantifier'
        assert isinstance(binary_quantifier, AggregativeQuantifier), \
            f'{self.binary_quantifier} does not seem to be of type Aggregative'
        self.binary_quantifier = binary_quantifier
        self.n_jobs = qp._get_njobs(n_jobs)
        self.parallel_backend = parallel_backend

    def classify(self, instances):
        """
        If the base quantifier is not probabilistic, returns a matrix of shape `(n,m,)` with `n` the number of
        instances and `m` the number of classes. The entry `(i,j)` is a binary value indicating whether instance
        `i `belongs to class `j`. The binary classifications are independent of each other, meaning that an instance
        can end up be attributed to 0, 1, or more classes.
        If the base quantifier is probabilistic, returns a matrix of shape `(n,m,2)` with `n` the number of instances
        and `m` the number of classes. The entry `(i,j,1)` (resp. `(i,j,0)`) is a value in [0,1] indicating the
        posterior probability that instance `i` belongs (resp. does not belong) to class `j`. The posterior
        probabilities are independent of each other, meaning that, in general, they do not sum up to one.

        :param instances: array-like
        :return: `np.ndarray`
        """

        classif_predictions = self._parallel(self._delayed_binary_classification, instances)
        if isinstance(self.binary_quantifier, AggregativeProbabilisticQuantifier):
            return np.swapaxes(classif_predictions, 0, 1)
        else:
            return classif_predictions.T

    def aggregate(self, classif_predictions):
        prevalences = self._parallel(self._delayed_binary_aggregate, classif_predictions)
        return F.normalize_prevalence(prevalences)

    def _delayed_binary_classification(self, c, X):
        return self.dict_binary_quantifiers[c].classify(X)

    def _delayed_binary_aggregate(self, c, classif_predictions):
        # the estimation for the positive class prevalence
        return self.dict_binary_quantifiers[c].aggregate(classif_predictions[:, c])[1]

