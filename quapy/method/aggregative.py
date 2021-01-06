import numpy as np
from copy import deepcopy
import functional as F
import error
from method.base import BaseQuantifier, BinaryQuantifier
from classification.svmperf import SVMperf
from data import LabelledCollection
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from joblib import Parallel, delayed
from abc import abstractmethod
from typing import Union


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
    def aggregate(self, classif_predictions:np.ndarray): ...

    def get_params(self, deep=True):
        return self.learner.get_params()

    def set_params(self, **parameters):
        self.learner.set_params(**parameters)

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def classes(self):
        return self.learner.classes_


class AggregativeProbabilisticQuantifier(AggregativeQuantifier):
    """
    Abstract class for quantification methods that base their estimations on the aggregation of posterior probabilities
    as returned by a probabilistic classifier. Aggregative Probabilistic Quantifiers thus extend Aggregative
    Quantifiers by implementing a _posterior_probabilities_ method returning values in [0,1] -- the posterior
    probabilities.
    """

    def posterior_probabilities(self, data):
        return self.learner.predict_proba(data)

    def quantify(self, instances):
        classif_posteriors = self.posterior_probabilities(instances)
        return self.aggregate(classif_posteriors)

    def set_params(self, **parameters):
        if isinstance(self.learner, CalibratedClassifierCV):
            parameters={'base_estimator__'+k:v for k,v in parameters.items()}
        self.learner.set_params(**parameters)



# Helper
# ------------------------------------
def training_helper(learner,
                    data: LabelledCollection,
                    fit_learner: bool = True,
                    ensure_probabilistic=False,
                    val_split:Union[LabelledCollection, float]=None):
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
                train, unused = data.split_stratified(train_prop=1-val_split)
            elif isinstance(val_split, LabelledCollection):
                train = data
                unused = val_split
            else:
                raise ValueError('train_val_split not understood; use either a float indicating the split proportion, '
                                 'or a LabelledCollection indicating the validation split')
        else:
            train, unused = data, None
        learner.fit(train.instances, train.labels)
    else:
        if ensure_probabilistic:
            if not hasattr(learner, 'predict_proba'):
                raise AssertionError('error: the learner cannot be calibrated since fit_learner is set to False')
        unused = data

    return learner, unused


# Methods
# ------------------------------------
class ClassifyAndCount(AggregativeQuantifier):
    """
    The most basic Quantification method. One that simply classifies all instances and countes how many have been
    attributed each of the classes in order to compute class prevalence estimates.
    """

    def __init__(self, learner):
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
        return F.prevalence_from_labels(classif_predictions, self.n_classes)


class AdjustedClassifyAndCount(AggregativeQuantifier):

    def __init__(self, learner):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True, val_split:Union[float, LabelledCollection]=0.3):
        """
        Trains a ACC quantifier
        :param data: the training set
        :param fit_learner: set to False to bypass the training (the learner is assumed to be already fit)
        :param val_split: either a float in (0,1) indicating the proportion of training instances to use for
         validation (e.g., 0.3 for using 30% of the training set as validation data), or a LabelledCollection
         indicating the validation set itself
        :return: self
        """
        self.learner, validation = training_helper(self.learner, data, fit_learner, val_split=val_split)
        self.cc = ClassifyAndCount(self.learner)
        y_ = self.classify(validation.instances)
        y  = validation.labels
        # estimate the matrix with entry (i,j) being the estimate of P(yi|yj), that is, the probability that a
        # document that belongs to yj ends up being classified as belonging to yi
        self.Pte_cond_estim_ = confusion_matrix(y,y_).T / validation.counts()
        return self

    def classify(self, data):
        return self.cc.classify(data)

    def aggregate(self, classif_predictions):
        prevs_estim = self.cc.aggregate(classif_predictions)
        return AdjustedClassifyAndCount.solve_adjustment(self.Pte_cond_estim_, prevs_estim)

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


class ProbabilisticClassifyAndCount(AggregativeProbabilisticQuantifier):
    def __init__(self, learner):
        self.learner = learner

    def fit(self, data : LabelledCollection, fit_learner=True):
        self.learner, _ = training_helper(self.learner, data, fit_learner, ensure_probabilistic=True)
        return self

    def aggregate(self, classif_posteriors):
        return F.prevalence_from_probabilities(classif_posteriors, binarize=False)


class ProbabilisticAdjustedClassifyAndCount(AggregativeProbabilisticQuantifier):

    def __init__(self, learner):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True, val_split:Union[float, LabelledCollection]=0.3):
        """
        Trains a PACC quantifier
        :param data: the training set
        :param fit_learner: set to False to bypass the training (the learner is assumed to be already fit)
        :param val_split: either a float in (0,1) indicating the proportion of training instances to use for
         validation (e.g., 0.3 for using 30% of the training set as validation data), or a LabelledCollection
         indicating the validation set itself
        :return: self
        """
        self.learner, validation = training_helper(
            self.learner, data, fit_learner, ensure_probabilistic=True, val_split=val_split
        )
        self.pcc = ProbabilisticClassifyAndCount(self.learner)
        y_ = self.soft_classify(validation.instances)
        y  = validation.labels
        confusion = np.empty(shape=(data.n_classes, data.n_classes))
        for yi in range(data.n_classes):
            confusion[yi] = y_[y==yi].mean(axis=0)

        self.Pte_cond_estim_ = confusion.T

        #y_ = self.classify(validation.instances)
        #y = validation.labels
        # estimate the matrix with entry (i,j) being the estimate of P(yi|yj), that is, the probability that a
        # document that belongs to yj ends up being classified as belonging to yi
        #self.Pte_cond_estim_ = confusion_matrix(y, y_).T / validation.counts()
        return self

    def aggregate(self, classif_posteriors):
        prevs_estim = self.pcc.aggregate(classif_posteriors)
        return AdjustedClassifyAndCount.solve_adjustment(self.Pte_cond_estim_, prevs_estim)

    def classify(self, data):
        return self.pcc.classify(data)

    def soft_classify(self, data):
        return self.pcc.posterior_probabilities(data)


class ExpectationMaximizationQuantifier(AggregativeProbabilisticQuantifier):

    MAX_ITER = 1000
    EPSILON = 1e-4

    def __init__(self, learner):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True):
        self.learner, _ = training_helper(self.learner, data, fit_learner, ensure_probabilistic=True)
        self.train_prevalence = F.prevalence_from_labels(data.labels, self.n_classes)
        return self

    def aggregate(self, classif_posteriors, epsilon=EPSILON):
        return self.EM(self.train_prevalence, classif_posteriors, epsilon)

    @classmethod
    def EM(cls, tr_prev, posterior_probabilities, epsilon=EPSILON):
        Px = posterior_probabilities
        Ptr = np.copy(tr_prev)
        qs = np.copy(Ptr)  # qs (the running estimate) is initialized as the training prevalence

        s, converged = 0, False
        qs_prev_ = None
        while not converged and s < ExpectationMaximizationQuantifier.MAX_ITER:
            # E-step: ps is Ps(y=+1|xi)
            ps_unnormalized = (qs / Ptr) * Px
            ps = ps_unnormalized / ps_unnormalized.sum(axis=1).reshape(-1,1)

            # M-step: qs_pos is Ps+1(y=+1)
            qs = ps.mean(axis=0)

            if qs_prev_ is not None and error.mae(qs, qs_prev_) < epsilon and s>10:
                converged = True

            qs_prev_ = qs
            s += 1

        if not converged:
            raise UserWarning('the method has reached the maximum number of iterations; it might have not converged')

        return qs


class HellingerDistanceY(AggregativeProbabilisticQuantifier, BinaryQuantifier):
    """
    Implementation of the method based on the Hellinger Distance y (HDy) proposed by
    González-Castro, V., Alaiz-Rodrı́guez, R., and Alegre, E. (2013). Class distribution
    estimation based on the Hellinger distance. Information Sciences, 218:146–164.
    """

    def __init__(self, learner):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True, val_split:Union[float, LabelledCollection]=0.3):
        """
        Trains a HDy quantifier
        :param data: the training set
        :param fit_learner: set to False to bypass the training (the learner is assumed to be already fit)
        :param val_split: either a float in (0,1) indicating the proportion of training instances to use for
         validation (e.g., 0.3 for using 30% of the training set as validation data), or a LabelledCollection
         indicating the validation set itself
        :return: self
        """
        self._check_binary(data, self.__class__.__name__)
        self.learner, validation = training_helper(
            self.learner, data, fit_learner, ensure_probabilistic=True, val_split=val_split)
        Px = self.posterior_probabilities(validation.instances)[:,1]  # takes only the P(y=+1|x)
        self.Pxy1 = Px[validation.labels == 1]
        self.Pxy0 = Px[validation.labels == 0]
        return self

    def aggregate(self, classif_posteriors):
        # "In this work, the number of bins b used in HDx and HDy was chosen from 10 to 110 in steps of 10,
        # and the final estimated a priori probability was taken as the median of these 11 estimates."
        # (González-Castro, et al., 2013).

        Px = classif_posteriors[:,1]  # takes only the P(y=+1|x)

        prev_estimations = []
        for bins in np.linspace(10, 110, 11, dtype=int):  #[10, 20, 30, ..., 100, 110]
            Pxy0_density, _ = np.histogram(self.Pxy0, bins=bins, range=(0, 1), density=True)
            Pxy1_density, _ = np.histogram(self.Pxy1, bins=bins, range=(0, 1), density=True)

            Px_test, _ = np.histogram(Px, bins=bins, range=(0, 1), density=True)

            prev_selected, min_dist = None, None
            for prev in F.prevalence_linspace(n_prevalences=100, repeat=1, smooth_limits_epsilon=0.0):
                Px_train = prev*Pxy1_density + (1 - prev)*Pxy0_density
                hdy = F.HellingerDistance(Px_train, Px_test)
                if prev_selected is None or hdy < min_dist:
                    prev_selected, min_dist = prev, hdy
            prev_estimations.append(prev_selected)

        pos_class_prev = np.median(prev_estimations)
        return np.asarray([1-pos_class_prev, pos_class_prev])


class ExplicitLossMinimisation(AggregativeQuantifier, BinaryQuantifier):

    def __init__(self, svmperf_base, loss, **kwargs):
        self.svmperf_base = svmperf_base
        self.loss = loss
        self.kwargs = kwargs

    def fit(self, data: LabelledCollection, fit_learner=True):
        self._check_binary(data, self.__class__.__name__)
        assert fit_learner, 'the method requires that fit_learner=True'
        self.learner = SVMperf(self.svmperf_base, loss=self.loss, **self.kwargs).fit(data.instances, data.labels)
        return self

    def aggregate(self, classif_predictions:np.ndarray):
        return F.prevalence_from_labels(classif_predictions, self.learner.n_classes_)

    def classify(self, X, y=None):
        return self.learner.predict(X)



class SVMQ(ExplicitLossMinimisation):
    def __init__(self, svmperf_base, **kwargs):
        super(SVMQ, self).__init__(svmperf_base, loss='q', **kwargs)


class SVMKLD(ExplicitLossMinimisation):
    def __init__(self, svmperf_base, **kwargs):
        super(SVMKLD, self).__init__(svmperf_base, loss='kld', **kwargs)


class SVMNKLD(ExplicitLossMinimisation):
    def __init__(self, svmperf_base, **kwargs):
        super(SVMNKLD, self).__init__(svmperf_base, loss='nkld', **kwargs)


class SVMAE(ExplicitLossMinimisation):
    def __init__(self, svmperf_base, **kwargs):
        super(SVMAE, self).__init__(svmperf_base, loss='mae', **kwargs)


class SVMRAE(ExplicitLossMinimisation):
    def __init__(self, svmperf_base, **kwargs):
        super(SVMRAE, self).__init__(svmperf_base, loss='mrae', **kwargs)


CC = ClassifyAndCount
ACC = AdjustedClassifyAndCount
PCC = ProbabilisticClassifyAndCount
PACC = ProbabilisticAdjustedClassifyAndCount
ELM = ExplicitLossMinimisation
EMQ = ExpectationMaximizationQuantifier
HDy = HellingerDistanceY


class OneVsAll(AggregativeQuantifier):
    """
    Allows any binary quantifier to perform quantification on single-label datasets. The method maintains one binary
    quantifier for each class, and then l1-normalizes the outputs so that the class prevelences sum up to 1.
    This variant was used, along with the ExplicitLossMinimization quantifier in
    Gao, W., Sebastiani, F.: From classification to quantification in tweet sentiment analysis.
    Social Network Analysis and Mining6(19), 1–22 (2016)
    """

    def __init__(self, binary_quantifier, n_jobs=-1):
        self.binary_quantifier = binary_quantifier
        self.n_jobs = n_jobs

    def fit(self, data: LabelledCollection, fit_learner=True):
        assert not data.binary, \
            f'{self.__class__.__name__} expect non-binary data'
        assert isinstance(self.binary_quantifier, BaseQuantifier), \
            f'{self.binary_quantifier} does not seem to be a Quantifier'
        assert fit_learner==True, 'fit_learner must be True'
        if not isinstance(self.binary_quantifier, BinaryQuantifier):
            raise ValueError(f'{self.binary_quantifier.__class__.__name__} does not seem to be an instance of '
                             f'{BinaryQuantifier.__class__.__name__}')
        self.dict_binary_quantifiers = {c: deepcopy(self.binary_quantifier) for c in data.classes_}
        self.__parallel(self._delayed_binary_fit, data)
        return self

    def classify(self, instances):
        classif_predictions_bin = self.__parallel(self._delayed_binary_classification, instances)
        return classif_predictions_bin.T

    def aggregate(self, classif_predictions_bin):
        assert set(np.unique(classif_predictions_bin)) == {0,1}, \
            'param classif_predictions_bin does not seem to be a valid matrix (ndarray) of binary ' \
            'predictions for each document (row) and class (columns)'
        prevalences = self.__parallel(self._delayed_binary_aggregate, classif_predictions_bin)
        #prevalences = []
        #for c in self.classes:
        #    prevalences.append(self._delayed_binary_aggregate(c, classif_predictions_bin))
        #prevalences = np.asarray(prevalences)
        return F.normalize_prevalence(prevalences)

    def quantify(self, X):
        prevalences = self.__parallel(self._delayed_binary_quantify, X)
        return F.normalize_prevalence(prevalences)

    def __parallel(self, func, *args, **kwargs):
        return np.asarray(
            Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(func)(c, *args, **kwargs) for c in self.classes
            )
        )

    @property
    def classes(self):
        return sorted(self.dict_binary_quantifiers.keys())

    def set_params(self, **parameters):
        self.binary_quantifier.set_params(**parameters)

    def get_params(self, deep=True):
        return self.binary_quantifier.get_params()

    def _delayed_binary_classification(self, c, X):
        return self.dict_binary_quantifiers[c].classify(X)

    def _delayed_binary_quantify(self, c, X):
        return self.dict_binary_quantifiers[c].quantify(X)[1]  # the estimation for the positive class prevalence

    def _delayed_binary_aggregate(self, c, classif_predictions):
        return self.dict_binary_quantifiers[c].aggregate(classif_predictions[:,c])[1]  # the estimation for the positive class prevalence

    def _delayed_binary_fit(self, c, data):
        bindata = LabelledCollection(data.instances, data.labels == c, n_classes=2)
        self.dict_binary_quantifiers[c].fit(bindata)


def isaggregative(model):
    return isinstance(model, AggregativeQuantifier)


def isprobabilistic(model):
    return isinstance(model, AggregativeProbabilisticQuantifier)


def isbinary(model):
    return isinstance(model, BinaryQuantifier)


