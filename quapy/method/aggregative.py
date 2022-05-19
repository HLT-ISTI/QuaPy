from abc import abstractmethod
from copy import deepcopy
from typing import Union

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import quapy as qp
import quapy.functional as F
from quapy.classification.svmperf import SVMperf
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier, BinaryQuantifier


# Abstract classes
# ------------------------------------

class AggregativeQuantifier(BaseQuantifier):
    """
    Abstract class for quantification methods that base their estimations on the aggregation of classification
    results. Aggregative Quantifiers thus implement a :meth:`classify` method and maintain a :attr:`learner` attribute.
    Subclasses of this abstract class must implement the method :meth:`aggregate` which computes the aggregation
    of label predictions. The method :meth:`quantify` comes with a default implementation based on
     :meth:`classify` and :meth:`aggregate`.
    """

    @abstractmethod
    def fit(self, data: LabelledCollection, fit_learner=True):
        """
        Trains the aggregative quantifier

        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
        :param fit_learner: whether or not to train the learner (default is True). Set to False if the
            learner has been trained outside the quantifier.
        :return: self
        """
        ...

    @property
    def learner(self):
        """
        Gives access to the classifier

        :return: the classifier (typically an sklearn's Estimator)
        """
        return self.learner_

    @learner.setter
    def learner(self, classifier):
        """
        Setter for the classifier

        :param classifier: the classifier
        """
        self.learner_ = classifier

    def classify(self, instances):
        """
        Provides the label predictions for the given instances.

        :param instances: array-like
        :return: np.ndarray of shape `(n_instances,)` with label predictions
        """
        return self.learner.predict(instances)

    def quantify(self, instances):
        """
        Generate class prevalence estimates for the sample's instances by aggregating the label predictions generated
        by the classifier.

        :param instances: array-like
        :return: `np.ndarray` of shape `(self.n_classes_,)` with class prevalence estimates.
        """
        classif_predictions = self.classify(instances)
        return self.aggregate(classif_predictions)

    @abstractmethod
    def aggregate(self, classif_predictions: np.ndarray):
        """
        Implements the aggregation of label predictions.

        :param classif_predictions: `np.ndarray` of label predictions
        :return: `np.ndarray` of shape `(self.n_classes_,)` with class prevalence estimates.
        """
        ...

    def get_params(self, deep=True):
        """
        Return the current parameters of the quantifier.

        :param deep: for compatibility with sklearn
        :return: a dictionary of param-value pairs
        """

        return self.learner.get_params()

    def set_params(self, **parameters):
        """
        Set the parameters of the quantifier.

        :param parameters: dictionary of param-value pairs
        """

        self.learner.set_params(**parameters)

    @property
    def classes_(self):
        """
        Class labels, in the same order in which class prevalence values are to be computed.
        This default implementation actually returns the class labels of the learner.

        :return: array-like
        """
        return self.learner.classes_

    @property
    def aggregative(self):
        """
        Returns True, indicating the quantifier is of type aggregative.

        :return: True
        """

        return True


class AggregativeProbabilisticQuantifier(AggregativeQuantifier):
    """
    Abstract class for quantification methods that base their estimations on the aggregation of posterior probabilities
    as returned by a probabilistic classifier. Aggregative Probabilistic Quantifiers thus extend Aggregative
    Quantifiers by implementing a _posterior_probabilities_ method returning values in [0,1] -- the posterior
    probabilities.
    """

    def posterior_probabilities(self, instances):
        return self.learner.predict_proba(instances)

    def predict_proba(self, instances):
        return self.posterior_probabilities(instances)

    def quantify(self, instances):
        classif_posteriors = self.posterior_probabilities(instances)
        return self.aggregate(classif_posteriors)

    def set_params(self, **parameters):
        if isinstance(self.learner, CalibratedClassifierCV):
            parameters = {'base_estimator__' + k: v for k, v in parameters.items()}
        self.learner.set_params(**parameters)

    @property
    def probabilistic(self):
        return True


# Helper
# ------------------------------------
def _training_helper(learner,
                     data: LabelledCollection,
                     fit_learner: bool = True,
                     ensure_probabilistic=False,
                     val_split: Union[LabelledCollection, float] = None):
    """
    Training procedure common to all Aggregative Quantifiers.

    :param learner: the learner to be fit
    :param data: the data on which to fit the learner. If requested, the data will be split before fitting the learner.
    :param fit_learner: whether or not to fit the learner (if False, then bypasses any action)
    :param ensure_probabilistic: if True, guarantees that the resulting classifier implements predict_proba (if the
        learner is not probabilistic, then a CalibratedCV instance of it is trained)
    :param val_split: if specified as a float, indicates the proportion of training instances that will define the
        validation split (e.g., 0.3 for using 30% of the training set as validation data); if specified as a
        LabelledCollection, represents the validation split itself
    :return: the learner trained on the training set, and the unused data (a _LabelledCollection_ if train_val_split>0
        or None otherwise) to be used as a validation set for any subsequent parameter fitting
    """
    if fit_learner:
        if ensure_probabilistic:
            if not hasattr(learner, 'predict_proba'):
                print(f'The learner {learner.__class__.__name__} does not seem to be probabilistic. '
                      f'The learner will be calibrated.')
                learner = CalibratedClassifierCV(learner, cv=5)
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

        if isinstance(learner, BaseQuantifier):
            learner.fit(train)
        else:
            learner.fit(*train.Xy)
    else:
        if ensure_probabilistic:
            if not hasattr(learner, 'predict_proba'):
                raise AssertionError('error: the learner cannot be calibrated since fit_learner is set to False')
        unused = None
        if isinstance(val_split, LabelledCollection):
            unused = val_split

    return learner, unused


# Methods
# ------------------------------------
class CC(AggregativeQuantifier):
    """
    The most basic Quantification method. One that simply classifies all instances and counts how many have been
    attributed to each of the classes in order to compute class prevalence estimates.

    :param learner: a sklearn's Estimator that generates a classifier
    """

    def __init__(self, learner: BaseEstimator):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True):
        """
        Trains the Classify & Count method unless `fit_learner` is False, in which case, the classifier is assumed to
        be already fit and there is nothing else to do.

        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
        :param fit_learner: if False, the classifier is assumed to be fit
        :return: self
        """
        self.learner, _ = _training_helper(self.learner, data, fit_learner)
        return self

    def aggregate(self, classif_predictions: np.ndarray):
        """
        Computes class prevalence estimates by counting the prevalence of each of the predicted labels.

        :param classif_predictions: array-like with label predictions
        :return: `np.ndarray` of shape `(self.n_classes_,)` with class prevalence estimates.
        """
        return F.prevalence_from_labels(classif_predictions, self.classes_)


class ACC(AggregativeQuantifier):
    """
    `Adjusted Classify & Count <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_,
    the "adjusted" variant of :class:`CC`, that corrects the predictions of CC
    according to the `misclassification rates`.

    :param learner: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, learner: BaseEstimator, val_split=0.4):
        self.learner = learner
        self.val_split = val_split

    def fit(self, data: LabelledCollection, fit_learner=True, val_split: Union[float, int, LabelledCollection] = None):
        """
        Trains a ACC quantifier.

        :param data: the training set
        :param fit_learner: set to False to bypass the training (the learner is assumed to be already fit)
        :param val_split: either a float in (0,1) indicating the proportion of training instances to use for
            validation (e.g., 0.3 for using 30% of the training set as validation data), or a LabelledCollection
            indicating the validation set itself, or an int indicating the number `k` of folds to be used in `k`-fold
            cross validation to estimate the parameters
        :return: self
        """
        if val_split is None:
            val_split = self.val_split
        if isinstance(val_split, int):
            assert fit_learner == True, \
                'the parameters for the adjustment cannot be estimated with kFCV with fit_learner=False'
            # kFCV estimation of parameters
            y, y_ = [], []
            kfcv = StratifiedKFold(n_splits=val_split)
            pbar = tqdm(kfcv.split(*data.Xy), total=val_split)
            for k, (training_idx, validation_idx) in enumerate(pbar):
                pbar.set_description(f'{self.__class__.__name__} fitting fold {k}')
                training = data.sampling_from_index(training_idx)
                validation = data.sampling_from_index(validation_idx)
                learner, val_data = _training_helper(self.learner, training, fit_learner, val_split=validation)
                y_.append(learner.predict(val_data.instances))
                y.append(val_data.labels)

            y = np.concatenate(y)
            y_ = np.concatenate(y_)
            class_count = data.counts()

            # fit the learner on all data
            self.learner, _ = _training_helper(self.learner, data, fit_learner, val_split=None)

        else:
            self.learner, val_data = _training_helper(self.learner, data, fit_learner, val_split=val_split)
            y_ = self.learner.predict(val_data.instances)
            y = val_data.labels

        self.cc = CC(self.learner)

        self.Pte_cond_estim_ = self.getPteCondEstim(data.classes_, y, y_)

        return self

    @classmethod
    def getPteCondEstim(cls, classes, y, y_):
        # estimate the matrix with entry (i,j) being the estimate of P(yi|yj), that is, the probability that a
        # document that belongs to yj ends up being classified as belonging to yi
        conf = confusion_matrix(y, y_, labels=classes).T
        conf = conf.astype(np.float)
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

    :param learner: a sklearn's Estimator that generates a classifier
    """

    def __init__(self, learner: BaseEstimator):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True):
        self.learner, _ = _training_helper(self.learner, data, fit_learner, ensure_probabilistic=True)
        return self

    def aggregate(self, classif_posteriors):
        return F.prevalence_from_probabilities(classif_posteriors, binarize=False)


class PACC(AggregativeProbabilisticQuantifier):
    """
    `Probabilistic Adjusted Classify & Count <https://ieeexplore.ieee.org/abstract/document/5694031>`_,
    the probabilistic variant of ACC that relies on the posterior probabilities returned by a probabilistic classifier.

    :param learner: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, learner: BaseEstimator, val_split=0.4):
        self.learner = learner
        self.val_split = val_split

    def fit(self, data: LabelledCollection, fit_learner=True, val_split: Union[float, int, LabelledCollection] = None):
        """
        Trains a PACC quantifier.

        :param data: the training set
        :param fit_learner: set to False to bypass the training (the learner is assumed to be already fit)
        :param val_split: either a float in (0,1) indicating the proportion of training instances to use for
         validation (e.g., 0.3 for using 30% of the training set as validation data), or a LabelledCollection
         indicating the validation set itself, or an int indicating the number k of folds to be used in kFCV
         to estimate the parameters
        :return: self
        """
        if val_split is None:
            val_split = self.val_split

        if isinstance(val_split, int):
            assert fit_learner == True, \
                'the parameters for the adjustment cannot be estimated with kFCV with fit_learner=False'
            # kFCV estimation of parameters
            y, y_ = [], []
            kfcv = StratifiedKFold(n_splits=val_split)
            pbar = tqdm(kfcv.split(*data.Xy), total=val_split)
            for k, (training_idx, validation_idx) in enumerate(pbar):
                pbar.set_description(f'{self.__class__.__name__} fitting fold {k}')
                training = data.sampling_from_index(training_idx)
                validation = data.sampling_from_index(validation_idx)
                learner, val_data = _training_helper(
                    self.learner, training, fit_learner, ensure_probabilistic=True, val_split=validation)
                y_.append(learner.predict_proba(val_data.instances))
                y.append(val_data.labels)

            y = np.concatenate(y)
            y_ = np.vstack(y_)

            # fit the learner on all data
            self.learner, _ = _training_helper(self.learner, data, fit_learner, ensure_probabilistic=True,
                                               val_split=None)
            classes = data.classes_

        else:
            self.learner, val_data = _training_helper(
                self.learner, data, fit_learner, ensure_probabilistic=True, val_split=val_split)
            y_ = self.learner.predict_proba(val_data.instances)
            y = val_data.labels
            classes = val_data.classes_

        self.pcc = PCC(self.learner)
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
    The `transform_prior` callback allows you to introduce ad-hoc regularizations which are not part of the
    original EMQ algorithm. This callback can, for instance, enhance or diminish small class prevalences if
    sparse or dense solutions should be promoted.

    The original method is described in:
    Saerens, M., Latinne, P., and Decaestecker, C. (2002).
    Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure.
    Neural Computation, 14(1): 21–41.

    :param learner: a sklearn's Estimator that generates a classifier
    :param transform_prior: an optional function :math:`R^c -> R^c` that transforms each intermediate estimate
    """

    MAX_ITER = 1000
    EPSILON = 1e-4

    def __init__(self, learner: BaseEstimator, transform_prior=None):
        self.learner = learner
        self.transform_prior = transform_prior

    def fit(self, data: LabelledCollection, fit_learner=True):
        self.learner, _ = _training_helper(self.learner, data, fit_learner, ensure_probabilistic=True)
        self.train_prevalence = F.prevalence_from_labels(data.labels, self.classes_)
        return self

    def aggregate(self, classif_posteriors, epsilon=EPSILON):
        priors, posteriors = self.EM(self.train_prevalence, classif_posteriors, epsilon, self.transform_prior)
        return priors

    def predict_proba(self, instances, epsilon=EPSILON):
        classif_posteriors = self.learner.predict_proba(instances)
        priors, posteriors = self.EM(self.train_prevalence, classif_posteriors, epsilon, self.transform_prior)
        return posteriors

    @classmethod
    def EM(cls, tr_prev, posterior_probabilities, epsilon=EPSILON, transform_prior=None):
        """
        Computes the `Expectation Maximization` routine.
        :param tr_prev: array-like, the training prevalence
        :param posterior_probabilities: `np.ndarray` of shape `(n_instances, n_classes,)` with the
            posterior probabilities
        :param epsilon: float, the threshold different between two consecutive iterations
            to reach before stopping the loop
        :param transform_prior: an optional function :math:`R^c -> R^c` that transforms each intermediate estimate
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

            # transformation of intermediate estimates
            if transform_prior is not None and not converged:
                qs = transform_prior(qs)

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

    :param learner: a sklearn's Estimator that generates a binary classifier
    :param val_split: a float in range (0,1) indicating the proportion of data to be used as a stratified held-out
        validation distribution, or a :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, learner: BaseEstimator, val_split=0.4):
        self.learner = learner
        self.val_split = val_split

    def fit(self, data: LabelledCollection, fit_learner=True, val_split: Union[float, LabelledCollection] = None):
        """
        Trains a HDy quantifier.

        :param data: the training set
        :param fit_learner: set to False to bypass the training (the learner is assumed to be already fit)
        :param val_split: either a float in (0,1) indicating the proportion of training instances to use for
         validation (e.g., 0.3 for using 30% of the training set as validation data), or a
         :class:`quapy.data.base.LabelledCollection` indicating the validation set itself
        :return: self
        """
        if val_split is None:
            val_split = self.val_split

        self._check_binary(data, self.__class__.__name__)
        self.learner, validation = _training_helper(
            self.learner, data, fit_learner, ensure_probabilistic=True, val_split=val_split)
        Px = self.posterior_probabilities(validation.instances)[:, 1]  # takes only the P(y=+1|x)
        self.Pxy1 = Px[validation.labels == self.learner.classes_[1]]
        self.Pxy0 = Px[validation.labels == self.learner.classes_[0]]
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


class ELM(AggregativeQuantifier, BinaryQuantifier):
    """
    Class of Explicit Loss Minimization (ELM) quantifiers.
    Quantifiers based on ELM represent a family of methods based on structured output learning;
    these quantifiers rely on classifiers that have been optimized using a quantification-oriented loss
    measure. This implementation relies on
    `Joachims’ SVM perf <https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html>`_ structured output
    learning algorithm, which has to be installed and patched for the purpose (see this
    `script <https://github.com/HLT-ISTI/QuaPy/blob/master/prepare_svmperf.sh>`_).

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`
    :param loss: the loss to optimize (see :attr:`quapy.classification.svmperf.SVMperf.valid_losses`)
    :param kwargs: rest of SVM perf's parameters
    """

    def __init__(self, svmperf_base=None, loss='01', **kwargs):
        self.svmperf_base = svmperf_base if svmperf_base is not None else qp.environ['SVMPERF_HOME']
        self.loss = loss
        self.kwargs = kwargs
        self.learner = SVMperf(self.svmperf_base, loss=self.loss, **self.kwargs)

    def fit(self, data: LabelledCollection, fit_learner=True):
        self._check_binary(data, self.__class__.__name__)
        assert fit_learner, 'the method requires that fit_learner=True'
        self.learner.fit(data.instances, data.labels)
        return self

    def aggregate(self, classif_predictions: np.ndarray):
        return F.prevalence_from_labels(classif_predictions, self.classes_)

    def classify(self, X, y=None):
        return self.learner.predict(X)


class SVMQ(ELM):
    """
    SVM(Q), which attempts to minimize the `Q` loss combining a classification-oriented loss and a
    quantification-oriented loss, as proposed by
    `Barranquero et al. 2015 <https://www.sciencedirect.com/science/article/pii/S003132031400291X>`_.
    Equivalent to:

    >>> ELM(svmperf_base, loss='q', **kwargs)

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`
    :param kwargs: rest of SVM perf's parameters
    """

    def __init__(self, svmperf_base=None, **kwargs):
        super(SVMQ, self).__init__(svmperf_base, loss='q', **kwargs)


class SVMKLD(ELM):
    """
    SVM(KLD), which attempts to minimize the Kullback-Leibler Divergence as proposed by
    `Esuli et al. 2015 <https://dl.acm.org/doi/abs/10.1145/2700406>`_.
    Equivalent to:

    >>> ELM(svmperf_base, loss='kld', **kwargs)

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`
    :param kwargs: rest of SVM perf's parameters
    """

    def __init__(self, svmperf_base=None, **kwargs):
        super(SVMKLD, self).__init__(svmperf_base, loss='kld', **kwargs)


class SVMNKLD(ELM):
    """
    SVM(NKLD), which attempts to minimize a version of the the Kullback-Leibler Divergence normalized
    via the logistic function, as proposed by
    `Esuli et al. 2015 <https://dl.acm.org/doi/abs/10.1145/2700406>`_.
    Equivalent to:

    >>> ELM(svmperf_base, loss='nkld', **kwargs)

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`
    :param kwargs: rest of SVM perf's parameters
    """

    def __init__(self, svmperf_base=None, **kwargs):
        super(SVMNKLD, self).__init__(svmperf_base, loss='nkld', **kwargs)


class SVMAE(ELM):
    """
    SVM(AE), which attempts to minimize Absolute Error as first used by
    `Moreo and Sebastiani, 2021 <https://arxiv.org/abs/2011.02552>`_.
    Equivalent to:

    >>> ELM(svmperf_base, loss='mae', **kwargs)

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`
    :param kwargs: rest of SVM perf's parameters
    """

    def __init__(self, svmperf_base=None, **kwargs):
        super(SVMAE, self).__init__(svmperf_base, loss='mae', **kwargs)


class SVMRAE(ELM):
    """
    SVM(RAE), which attempts to minimize Relative Absolute Error as first used by
    `Moreo and Sebastiani, 2021 <https://arxiv.org/abs/2011.02552>`_.
    Equivalent to:

    >>> ELM(svmperf_base, loss='mrae', **kwargs)

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`
    :param kwargs: rest of SVM perf's parameters
    """

    def __init__(self, svmperf_base=None, **kwargs):
        super(SVMRAE, self).__init__(svmperf_base, loss='mrae', **kwargs)


class ThresholdOptimization(AggregativeQuantifier, BinaryQuantifier):
    """
    Abstract class of Threshold Optimization variants for :class:`ACC` as proposed by
    `Forman 2006 <https://dl.acm.org/doi/abs/10.1145/1150402.1150423>`_ and
    `Forman 2008 <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_.
    The goal is to bring improved stability to the denominator of the adjustment.
    The different variants are based on different heuristics for choosing a decision threshold
    that would allow for more true positives and many more false positives, on the grounds this
    would deliver larger denominators.

    :param learner: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, learner: BaseEstimator, val_split=0.4):
        self.learner = learner
        self.val_split = val_split

    def fit(self, data: LabelledCollection, fit_learner=True, val_split: Union[float, int, LabelledCollection] = None):
        self._check_binary(data, "Threshold Optimization")

        if val_split is None:
            val_split = self.val_split
        if isinstance(val_split, int):
            assert fit_learner == True, \
                'the parameters for the adjustment cannot be estimated with kFCV with fit_learner=False'
            # kFCV estimation of parameters
            y, probabilities = [], []
            kfcv = StratifiedKFold(n_splits=val_split)
            pbar = tqdm(kfcv.split(*data.Xy), total=val_split)
            for k, (training_idx, validation_idx) in enumerate(pbar):
                pbar.set_description(f'{self.__class__.__name__} fitting fold {k}')
                training = data.sampling_from_index(training_idx)
                validation = data.sampling_from_index(validation_idx)
                learner, val_data = _training_helper(self.learner, training, fit_learner, val_split=validation)
                probabilities.append(learner.predict_proba(val_data.instances))
                y.append(val_data.labels)

            y = np.concatenate(y)
            probabilities = np.concatenate(probabilities)

            # fit the learner on all data
            self.learner, _ = _training_helper(self.learner, data, fit_learner, val_split=None)

        else:
            self.learner, val_data = _training_helper(self.learner, data, fit_learner, val_split=val_split)
            probabilities = self.learner.predict_proba(val_data.instances)
            y = val_data.labels

        self.cc = CC(self.learner)

        self.tpr, self.fpr = self._optimize_threshold(y, probabilities)

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
            return 0
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

    :param learner: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, learner: BaseEstimator, val_split=0.4):
        super().__init__(learner, val_split)

    def _condition(self, tpr, fpr) -> float:
        return abs(tpr - 0.5)


class MAX(ThresholdOptimization):
    """
    Threshold Optimization variant for :class:`ACC` as proposed by
    `Forman 2006 <https://dl.acm.org/doi/abs/10.1145/1150402.1150423>`_ and
    `Forman 2008 <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_ that looks
    for the threshold that maximizes `tpr-fpr`.
    The goal is to bring improved stability to the denominator of the adjustment.

    :param learner: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, learner: BaseEstimator, val_split=0.4):
        super().__init__(learner, val_split)

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

    :param learner: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, learner: BaseEstimator, val_split=0.4):
        super().__init__(learner, val_split)

    def _condition(self, tpr, fpr) -> float:
        return abs(1 - (tpr + fpr))


class MS(ThresholdOptimization):
    """
    Median Sweep. Threshold Optimization variant for :class:`ACC` as proposed by
    `Forman 2006 <https://dl.acm.org/doi/abs/10.1145/1150402.1150423>`_ and
    `Forman 2008 <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_ that generates
    class prevalence estimates for all decision thresholds and returns the median of them all.
    The goal is to bring improved stability to the denominator of the adjustment.

    :param learner: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """
    def __init__(self, learner: BaseEstimator, val_split=0.4):
        super().__init__(learner, val_split)

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

    :param learner: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1, default 0.4), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """
    def __init__(self, learner: BaseEstimator, val_split=0.4):
        super().__init__(learner, val_split)

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
ExplicitLossMinimisation = ELM
MedianSweep = MS
MedianSweep2 = MS2


class OneVsAll(AggregativeQuantifier):
    """
    Allows any binary quantifier to perform quantification on single-label datasets.
    The method maintains one binary quantifier for each class, and then l1-normalizes the outputs so that the
    class prevelences sum up to 1.
    This variant was used, along with the :class:`EMQ` quantifier, in
    `Gao and Sebastiani, 2016 <https://link.springer.com/content/pdf/10.1007/s13278-016-0327-z.pdf>`_.

    :param learner: a sklearn's Estimator that generates a binary classifier
    :param n_jobs: number of parallel workers
    """

    def __init__(self, binary_quantifier, n_jobs=-1):
        self.binary_quantifier = binary_quantifier
        self.n_jobs = n_jobs

    def fit(self, data: LabelledCollection, fit_learner=True):
        assert not data.binary, \
            f'{self.__class__.__name__} expect non-binary data'
        assert isinstance(self.binary_quantifier, BaseQuantifier), \
            f'{self.binary_quantifier} does not seem to be a Quantifier'
        assert fit_learner == True, 'fit_learner must be True'

        self.dict_binary_quantifiers = {c: deepcopy(self.binary_quantifier) for c in data.classes_}
        self.__parallel(self._delayed_binary_fit, data)
        return self

    def classify(self, instances):
        """
        Returns a matrix of shape `(n,m,)` with `n` the number of instances and `m` the number of classes. The entry
        `(i,j)` is a binary value indicating whether instance `i `belongs to class `j`. The binary classifications are
        independent of each other, meaning that an instance can end up be attributed to 0, 1, or more classes.

        :param instances: array-like
        :return: `np.ndarray`
        """

        classif_predictions_bin = self.__parallel(self._delayed_binary_classification, instances)
        return classif_predictions_bin.T

    def posterior_probabilities(self, instances):
        """
        Returns a matrix of shape `(n,m,2)` with `n` the number of instances and `m` the number of classes. The entry
        `(i,j,1)` (resp. `(i,j,0)`) is a value in [0,1] indicating the posterior probability that instance `i` belongs
        (resp. does not belong) to class `j`.
        The posterior probabilities are independent of each other, meaning that, in general, they do not sum
        up to one.

        :param instances: array-like
        :return: `np.ndarray`
        """

        if not self.binary_quantifier.probabilistic:
            raise NotImplementedError(f'{self.__class__.__name__} does not implement posterior_probabilities because '
                                      f'the base quantifier {self.binary_quantifier.__class__.__name__} is not '
                                      f'probabilistic')
        posterior_predictions_bin = self.__parallel(self._delayed_binary_posteriors, instances)
        return np.swapaxes(posterior_predictions_bin, 0, 1)

    def aggregate(self, classif_predictions_bin):
        if self.probabilistic:
            assert classif_predictions_bin.shape[1] == self.n_classes and classif_predictions_bin.shape[2] == 2, \
                'param classif_predictions_bin does not seem to be a valid matrix (ndarray) of posterior ' \
                'probabilities (2 dimensions) for each document (row) and class (columns)'
        else:
            assert set(np.unique(classif_predictions_bin)).issubset({0, 1}), \
                'param classif_predictions_bin does not seem to be a valid matrix (ndarray) of binary ' \
                'predictions for each document (row) and class (columns)'
        prevalences = self.__parallel(self._delayed_binary_aggregate, classif_predictions_bin)
        return F.normalize_prevalence(prevalences)

    def quantify(self, X):
        if self.probabilistic:
            predictions = self.posterior_probabilities(X)
        else:
            predictions = self.classify(X)
        return self.aggregate(predictions)

    def __parallel(self, func, *args, **kwargs):
        return np.asarray(
            # some quantifiers (in particular, ELM-based ones) cannot be run with multiprocess, since the temp dir they
            # create during the fit will be removed and be no longer available for the predict...
            Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(func)(c, *args, **kwargs) for c in self.classes_
            )
        )

    @property
    def classes_(self):
        return sorted(self.dict_binary_quantifiers.keys())

    def set_params(self, **parameters):
        self.binary_quantifier.set_params(**parameters)

    def get_params(self, deep=True):
        return self.binary_quantifier.get_params()

    def _delayed_binary_classification(self, c, X):
        return self.dict_binary_quantifiers[c].classify(X)

    def _delayed_binary_posteriors(self, c, X):
        return self.dict_binary_quantifiers[c].posterior_probabilities(X)

    def _delayed_binary_aggregate(self, c, classif_predictions):
        # the estimation for the positive class prevalence
        return self.dict_binary_quantifiers[c].aggregate(classif_predictions[:, c])[1]

    def _delayed_binary_fit(self, c, data):
        bindata = LabelledCollection(data.instances, data.labels == c, classes_=[False, True])
        self.dict_binary_quantifiers[c].fit(bindata)

    @property
    def binary(self):
        """
        Informs that the classifier is not binary

        :return: False
        """
        return False

    @property
    def probabilistic(self):
        """
        Indicates if the classifier is probabilistic or not (depending on the nature of the base classifier).

        :return: boolean
        """

        return self.binary_quantifier.probabilistic