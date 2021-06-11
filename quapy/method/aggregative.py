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
    results. Aggregative Quantifiers thus implement a _classify_ method and maintain a _learner_ attribute.
    """

    @abstractmethod
    def fit(self, data: LabelledCollection, fit_learner=True): ...

    @property
    def learner(self):
        return self.learner_

    @learner.setter
    def learner(self, value):
        self.learner_ = value

    def classify(self, instances):
        return self.learner.predict(instances)

    def quantify(self, instances):
        classif_predictions = self.classify(instances)
        return self.aggregate(classif_predictions)

    @abstractmethod
    def aggregate(self, classif_predictions: np.ndarray): ...

    def get_params(self, deep=True):
        return self.learner.get_params()

    def set_params(self, **parameters):
        self.learner.set_params(**parameters)

    @property
    def n_classes(self):
        return len(self.classes_)

    @property
    def classes_(self):
        return self.learner.classes_

    @property
    def aggregative(self):
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
def training_helper(learner,
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
            elif val_split.__class__.__name__ == LabelledCollection.__name__:  # isinstance(val_split, LabelledCollection):
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
            learner.fit(train.instances, train.labels)
    else:
        if ensure_probabilistic:
            if not hasattr(learner, 'predict_proba'):
                raise AssertionError('error: the learner cannot be calibrated since fit_learner is set to False')
        unused = data

    return learner, unused


# Methods
# ------------------------------------
class CC(AggregativeQuantifier):
    """
    The most basic Quantification method. One that simply classifies all instances and countes how many have been
    attributed each of the classes in order to compute class prevalence estimates.
    """

    def __init__(self, learner: BaseEstimator):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True):
        """
        Trains the Classify & Count method unless _fit_learner_ is False, in which case it is assumed to be already fit.
        :param data: training data
        :param fit_learner: if False, the classifier is assumed to be fit
        :return: self
        """
        self.learner, _ = training_helper(self.learner, data, fit_learner)
        return self

    def aggregate(self, classif_predictions):
        return F.prevalence_from_labels(classif_predictions, self.classes_)


class ACC(AggregativeQuantifier):

    def __init__(self, learner: BaseEstimator, val_split=0.4):
        self.learner = learner
        self.val_split = val_split

    def fit(self, data: LabelledCollection, fit_learner=True, val_split: Union[float, int, LabelledCollection] = None):
        """
        Trains a ACC quantifier
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
            # kFCV estimation of parameters
            y, y_ = [], []
            kfcv = StratifiedKFold(n_splits=val_split)
            pbar = tqdm(kfcv.split(*data.Xy), total=val_split)
            for k, (training_idx, validation_idx) in enumerate(pbar):
                pbar.set_description(f'{self.__class__.__name__} fitting fold {k}')
                training = data.sampling_from_index(training_idx)
                validation = data.sampling_from_index(validation_idx)
                learner, val_data = training_helper(self.learner, training, fit_learner, val_split=validation)
                y_.append(learner.predict(val_data.instances))
                y.append(val_data.labels)

            y = np.concatenate(y)
            y_ = np.concatenate(y_)
            class_count = data.counts()

            # fit the learner on all data
            self.learner, _ = training_helper(self.learner, data, fit_learner, val_split=None)

        else:
            self.learner, val_data = training_helper(self.learner, data, fit_learner, val_split=val_split)
            y_ = self.learner.predict(val_data.instances)
            y = val_data.labels
            class_count = val_data.counts()

        self.cc = CC(self.learner)

        # estimate the matrix with entry (i,j) being the estimate of P(yi|yj), that is, the probability that a
        # document that belongs to yj ends up being classified as belonging to yi
        self.Pte_cond_estim_ = confusion_matrix(y, y_).T / class_count

        return self

    def classify(self, data):
        return self.cc.classify(data)

    def aggregate(self, classif_predictions):
        prevs_estim = self.cc.aggregate(classif_predictions)
        return ACC.solve_adjustment(self.Pte_cond_estim_, prevs_estim)

    @classmethod
    def solve_adjustment(cls, PteCondEstim, prevs_estim):
        # solve for the linear system Ax = B with A=PteCondEstim and B = prevs_estim
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
    def __init__(self, learner: BaseEstimator):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True):
        self.learner, _ = training_helper(self.learner, data, fit_learner, ensure_probabilistic=True)
        return self

    def aggregate(self, classif_posteriors):
        return F.prevalence_from_probabilities(classif_posteriors, binarize=False)


class PACC(AggregativeProbabilisticQuantifier):

    def __init__(self, learner: BaseEstimator, val_split=0.4):
        self.learner = learner
        self.val_split = val_split

    def fit(self, data: LabelledCollection, fit_learner=True, val_split: Union[float, int, LabelledCollection] = None):
        """
        Trains a PACC quantifier
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
            # kFCV estimation of parameters
            y, y_ = [], []
            kfcv = StratifiedKFold(n_splits=val_split)
            pbar = tqdm(kfcv.split(*data.Xy), total=val_split)
            for k, (training_idx, validation_idx) in enumerate(pbar):
                pbar.set_description(f'{self.__class__.__name__} fitting fold {k}')
                training = data.sampling_from_index(training_idx)
                validation = data.sampling_from_index(validation_idx)
                learner, val_data = training_helper(
                    self.learner, training, fit_learner, ensure_probabilistic=True, val_split=validation)
                y_.append(learner.predict_proba(val_data.instances))
                y.append(val_data.labels)

            y = np.concatenate(y)
            y_ = np.vstack(y_)

            # fit the learner on all data
            self.learner, _ = training_helper(self.learner, data, fit_learner, ensure_probabilistic=True,
                                              val_split=None)

        else:
            self.learner, val_data = training_helper(
                self.learner, data, fit_learner, ensure_probabilistic=True, val_split=val_split)
            y_ = self.learner.predict_proba(val_data.instances)
            y = val_data.labels

        self.pcc = PCC(self.learner)

        # estimate the matrix with entry (i,j) being the estimate of P(yi|yj), that is, the probability that a
        # document that belongs to yj ends up being classified as belonging to yi
        confusion = np.empty(shape=(data.n_classes, data.n_classes))
        for i,class_ in enumerate(data.classes_):
            confusion[i] = y_[y == class_].mean(axis=0)

        self.Pte_cond_estim_ = confusion.T

        return self

    def aggregate(self, classif_posteriors):
        prevs_estim = self.pcc.aggregate(classif_posteriors)
        return ACC.solve_adjustment(self.Pte_cond_estim_, prevs_estim)

    def classify(self, data):
        return self.pcc.classify(data)


class EMQ(AggregativeProbabilisticQuantifier):
    """
    The method is described in:
    Saerens, M., Latinne, P., and Decaestecker, C. (2002).
    Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure.
    Neural Computation, 14(1): 21–41.
    """

    MAX_ITER = 1000
    EPSILON = 1e-4

    def __init__(self, learner: BaseEstimator):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True):
        self.learner, _ = training_helper(self.learner, data, fit_learner, ensure_probabilistic=True)
        self.train_prevalence = F.prevalence_from_labels(data.labels, self.classes_)
        return self

    def aggregate(self, classif_posteriors, epsilon=EPSILON):
        priors, posteriors = self.EM(self.train_prevalence, classif_posteriors, epsilon)
        return priors

    def predict_proba(self, instances, epsilon=EPSILON):
        classif_posteriors = self.learner.predict_proba(instances)
        priors, posteriors = self.EM(self.train_prevalence, classif_posteriors, epsilon)
        return posteriors

    @classmethod
    def EM(cls, tr_prev, posterior_probabilities, epsilon=EPSILON):
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
    Implementation of the method based on the Hellinger Distance y (HDy) proposed by
    González-Castro, V., Alaiz-Rodrı́guez, R., and Alegre, E. (2013). Class distribution
    estimation based on the Hellinger distance. Information Sciences, 218:146–164.
    """

    def __init__(self, learner: BaseEstimator, val_split=0.4):
        self.learner = learner
        self.val_split = val_split

    def fit(self, data: LabelledCollection, fit_learner=True, val_split: Union[float, LabelledCollection] = None):
        """
        Trains a HDy quantifier
        :param data: the training set
        :param fit_learner: set to False to bypass the training (the learner is assumed to be already fit)
        :param val_split: either a float in (0,1) indicating the proportion of training instances to use for
         validation (e.g., 0.3 for using 30% of the training set as validation data), or a LabelledCollection
         indicating the validation set itself
        :return: self
        """
        if val_split is None:
            val_split = self.val_split

        self._check_binary(data, self.__class__.__name__)
        self.learner, validation = training_helper(
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
            for prev in F.prevalence_linspace(n_prevalences=100, repeat=1, smooth_limits_epsilon=0.0):
                Px_train = prev * Pxy1_density + (1 - prev) * Pxy0_density
                hdy = F.HellingerDistance(Px_train, Px_test)
                if prev_selected is None or hdy < min_dist:
                    prev_selected, min_dist = prev, hdy
            prev_estimations.append(prev_selected)

        class1_prev = np.median(prev_estimations)
        return np.asarray([1 - class1_prev, class1_prev])


class ELM(AggregativeQuantifier, BinaryQuantifier):

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
    Barranquero, J., Díez, J., and del Coz, J. J. (2015).
    Quantification-oriented learning based on reliable classifiers.
    Pattern Recognition, 48(2):591–604.
    """

    def __init__(self, svmperf_base=None, **kwargs):
        super(SVMQ, self).__init__(svmperf_base, loss='q', **kwargs)


class SVMKLD(ELM):
    """
    Esuli, A. and Sebastiani, F. (2015).
    Optimizing text quantifiers for multivariate loss functions.
    ACM Transactions on Knowledge Discovery and Data, 9(4):Article 27.
    """

    def __init__(self, svmperf_base=None, **kwargs):
        super(SVMKLD, self).__init__(svmperf_base, loss='kld', **kwargs)


class SVMNKLD(ELM):
    """
    Esuli, A. and Sebastiani, F. (2015).
    Optimizing text quantifiers for multivariate loss functions.
    ACM Transactions on Knowledge Discovery and Data, 9(4):Article 27.
    """

    def __init__(self, svmperf_base=None, **kwargs):
        super(SVMNKLD, self).__init__(svmperf_base, loss='nkld', **kwargs)


class SVMAE(ELM):
    def __init__(self, svmperf_base=None, **kwargs):
        super(SVMAE, self).__init__(svmperf_base, loss='mae', **kwargs)


class SVMRAE(ELM):
    def __init__(self, svmperf_base=None, **kwargs):
        super(SVMRAE, self).__init__(svmperf_base, loss='mrae', **kwargs)


ClassifyAndCount = CC
AdjustedClassifyAndCount = ACC
ProbabilisticClassifyAndCount = PCC
ProbabilisticAdjustedClassifyAndCount = PACC
ExpectationMaximizationQuantifier = EMQ
HellingerDistanceY = HDy
ExplicitLossMinimisation = ELM


class OneVsAll(AggregativeQuantifier):
    """
    Allows any binary quantifier to perform quantification on single-label datasets. The method maintains one binary
    quantifier for each class, and then l1-normalizes the outputs so that the class prevelences sum up to 1.
    This variant was used, along with the ExplicitLossMinimization quantifier in
    Gao, W., Sebastiani, F.: From classification to quantification in tweet sentiment analysis.
    Social Network Analysis and Mining 6(19), 1–22 (2016)
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
        # returns a matrix of shape (n,m) with n the number of instances and m the number of classes. The entry
        # (i,j) is a binary value indicating whether instance i belongs to class j. The binary classifications are
        # independent of each other, meaning that an instance can end up be attributed to 0, 1, or more classes.
        classif_predictions_bin = self.__parallel(self._delayed_binary_classification, instances)
        return classif_predictions_bin.T

    def posterior_probabilities(self, instances):
        # returns a matrix of shape (n,m,2) with n the number of instances and m the number of classes. The entry
        # (i,j,1) (resp. (i,j,0)) is a value in [0,1] indicating the posterior probability that instance i belongs
        # (resp. does not belong) to class j.
        # The posterior probabilities are independent of each other, meaning that, in general, they do not sum
        # up to one.
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
        return False

    @property
    def probabilistic(self):
        return self.binary_quantifier.probabilistic
