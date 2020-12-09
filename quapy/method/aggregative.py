import numpy as np
from copy import deepcopy
import functional as F
import error
from method.base import BaseQuantifier
from quapy.classification.svmperf import SVMperf
from quapy.data import LabelledCollection
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from joblib import Parallel, delayed
from abc import abstractmethod


# Abstract classes
# ------------------------------------

class AggregativeQuantifier(BaseQuantifier):
    """
    Abstract class for quantification methods that base their estimations on the aggregation of classification
    results. Aggregative Quantifiers thus implement a _classify_ method and maintain a _learner_ attribute.
    """

    @abstractmethod
    def fit(self, data: LabelledCollection, fit_learner=True, *args): ...

    @property
    def learner(self):
        return self.learner_

    @learner.setter
    def learner(self, value):
        self.learner_ = value

    def classify(self, instances):
        return self.learner.predict(instances)

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
    Quantifiersimplement by implementing a _soft_classify_ method returning values in [0,1] -- the posterior
    probabilities.
    """

    def soft_classify(self, data):
        return self.learner.predict_proba(data)

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
                    train_val_split=None):
    """
    Training procedure common to all Aggregative Quantifiers.
    :param learner: the learner to be fit
    :param data: the data on which to fit the learner. If requested, the data will be split before fitting the learner.
    :param fit_learner: whether or not to fit the learner (if False, then bypasses any action)
    :param ensure_probabilistic: if True, guarantees that the resulting classifier implements predict_proba (if the
    learner is not probabilistic, then a CalibratedCV instance of it is trained)
    :param train_val_split: if specified, indicates the proportion of training instances on which to fit the learner
    :return: the learner trained on the training set, and the unused data (a _LabelledCollection_ if train_val_split>0
    or None otherwise) to be used as a validation set for any subsequent parameter fitting
    """
    if fit_learner:
        if ensure_probabilistic:
            if not hasattr(learner, 'predict_proba'):
                print(f'The learner {learner.__class__.__name__} does not seem to be probabilistic. '
                      f'The learner will be calibrated.')
                learner = CalibratedClassifierCV(learner, cv=5)
        if train_val_split is not None:
            if not (0 < train_val_split < 1):
                raise ValueError(f'train/val split {train_val_split} out of range, must be in (0,1)')
            train, unused = data.split_stratified(train_prop=train_val_split)
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

    def fit(self, data: LabelledCollection, fit_learner=True, *args):
        """
        Trains the Classify & Count method unless _fit_learner_ is False, in which case it is assumed to be already fit.
        :param data: training data
        :param fit_learner: if False, the classifier is assumed to be fit
        :param args: unused
        :return: self
        """
        self.learner, _ = training_helper(self.learner, data, fit_learner)
        return self

    def quantify(self, instances, *args):
        classification = self.classify(instances)  # classify
        return F.prevalence_from_labels(classification, self.n_classes)  # & count


class AdjustedClassifyAndCount(AggregativeQuantifier):

    def __init__(self, learner):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True, train_val_split=0.6):
        self.learner, validation = training_helper(self.learner, data, fit_learner, train_val_split=train_val_split)
        self.cc = ClassifyAndCount(self.learner)
        y_ = self.cc.classify(validation.instances)
        y  = validation.labels
        # estimate the matrix with entry (i,j) being the estimate of P(yi|yj), that is, the probability that a
        # document that belongs to yj ends up being classified as belonging to yi
        self.Pte_cond_estim_ = confusion_matrix(y,y_).T / validation.counts()
        return self

    def quantify(self, instances, *args):
        prevs_estim = self.cc.quantify(instances)
        # solve for the linear system Ax = B with A=Pte_cond_estim and B = prevs_estim
        A = self.Pte_cond_estim_
        B = prevs_estim
        try:
            adjusted_prevs = np.linalg.solve(A, B)
            adjusted_prevs = np.clip(adjusted_prevs, 0, 1)
            adjusted_prevs /= adjusted_prevs.sum()
        except np.linalg.LinAlgError:
            adjusted_prevs = prevs_estim  # no way to adjust them!
        return adjusted_prevs

    def classify(self, data):
        return self.cc.classify(data)


class ProbabilisticClassifyAndCount(AggregativeProbabilisticQuantifier):
    def __init__(self, learner):
        self.learner = learner

    def fit(self, data : LabelledCollection, fit_learner=True, *args):
        self.learner, _ = training_helper(self.learner, data, fit_learner, ensure_probabilistic=True)
        return self

    def quantify(self, instances, *args):
        posteriors = self.soft_classify(instances)  # classify
        prevalences = F.prevalence_from_probabilities(posteriors, binarize=False)  # & count
        return prevalences


class ProbabilisticAdjustedClassifyAndCount(AggregativeQuantifier):

    def __init__(self, learner):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True, train_val_split=0.6):
        self.learner, validation = training_helper(
            self.learner, data, fit_learner, ensure_probabilistic=True, train_val_split=train_val_split
        )
        self.pcc = ProbabilisticClassifyAndCount(self.learner)
        y_ = self.pcc.classify(validation.instances)
        y = validation.labels
        # estimate the matrix with entry (i,j) being the estimate of P(yi|yj), that is, the probability that a
        # document that belongs to yj ends up being classified as belonging to yi
        self.Pte_cond_estim_ = confusion_matrix(y, y_).T / validation.counts()
        return self

    def quantify(self, instances, *args):
        prevs_estim = self.pcc.quantify(instances)
        A = self.Pte_cond_estim_
        B = prevs_estim
        try:
            adjusted_prevs = np.linalg.solve(A, B)
            adjusted_prevs = np.clip(adjusted_prevs, 0, 1)
            adjusted_prevs /= adjusted_prevs.sum()
        except np.linalg.LinAlgError:
            adjusted_prevs = prevs_estim  # no way to adjust them!
        return adjusted_prevs

    def classify(self, data):
        return self.pcc.classify(data)


class ExpectationMaximizationQuantifier(AggregativeProbabilisticQuantifier):

    MAX_ITER = 1000
    EPSILON = 1e-4

    def __init__(self, learner, verbose=False):
        self.learner = learner
        self.verbose = verbose

    def fit(self, data: LabelledCollection, fit_learner=True, *args):
        self.learner, _ = training_helper(self.learner, data, fit_learner, ensure_probabilistic=True)
        self.train_prevalence = F.prevalence_from_labels(data.labels, self.n_classes)
        return self

    def quantify(self, X, epsilon=EPSILON):
        tr_prev=self.train_prevalence
        posteriors = self.soft_classify(X)
        return self.EM(tr_prev, posteriors, self.verbose, epsilon)

    @classmethod
    def EM(cls, tr_prev, posterior_probabilities, verbose=False, epsilon=EPSILON):
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

        if verbose:
            print('-'*80)

        if not converged:
            raise UserWarning('the method has reached the maximum number of iterations; it might have not converged')

        return qs


class HellingerDistanceY(AggregativeProbabilisticQuantifier):
    """
    Implementation of the method based on the Hellinger Distance y (HDy) proposed by
    González-Castro, V., Alaiz-Rodrı́guez, R., and Alegre, E. (2013). Class distribution
    estimation based on the Hellinger distance. Information Sciences, 218:146–164.
    """

    def __init__(self, learner):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True, train_val_split=0.6):
        assert data.binary, f'{self.__class__.__name__} works only on problems of binary classification. ' \
                            f'Use the class OneVsAll to enable {self.__class__.__name__} work on single-label data.'
        self.learner, validation = training_helper(
            self.learner, data, fit_learner, ensure_probabilistic=True, train_val_split=train_val_split)
        Px = self.soft_classify(validation.instances)
        self.Pxy1 = Px[validation.labels == 1]
        self.Pxy0 = Px[validation.labels == 0]
        return self

    def quantify(self, instances, *args):
        # "In this work, the number of bins b used in HDx and HDy was chosen from 10 to 110 in steps of 10,
        # and the final estimated a priori probability was taken as the median of these 11 estimates."
        # (González-Castro, et al., 2013).

        Px = self.soft_classify(instances)

        prev_estimations = []
        for bins in np.linspace(10, 110, 11, dtype=int): #[10, 20, 30, ..., 100, 110]
            Pxy0_density, _ = np.histogram(self.Pxy0, bins=bins, range=(0, 1), density=True)
            Pxy1_density, _ = np.histogram(self.Pxy1, bins=bins, range=(0, 1), density=True)

            Px_test, _ = np.histogram(Px, bins=bins, range=(0, 1), density=True)

            prev_selected, min_dist = None, None
            for prev in F.prevalence_linspace(n_prevalences=100, repeat=1, smooth_limits_epsilon=0.0):
                Px_train = prev*Pxy1_density + (1 - prev)*Pxy0_density
                hdy = HellingerDistanceY.HellingerDistance(Px_train, Px_test)
                if prev_selected is None or hdy < min_dist:
                    prev_selected, min_dist = prev, hdy
            prev_estimations.append(prev_selected)

        pos_class_prev = np.median(prev_estimations)
        return np.asarray([1-pos_class_prev, pos_class_prev])

    @classmethod
    def HellingerDistance(cls, P, Q):
        return np.sqrt(np.sum((np.sqrt(P) - np.sqrt(Q))**2))


class OneVsAll(AggregativeQuantifier):
    """
    Allows any binary quantifier to perform quantification on single-label datasets. The method maintains one binary
    quantifier for each class, and then l1-normalizes the outputs so that the class prevelences sum up to 1.
    """

    def __init__(self, binary_method, n_jobs=-1):
        self.binary_method = binary_method
        self.n_jobs = n_jobs

    def fit(self, data: LabelledCollection, **kwargs):
        assert not data.binary, f'{self.__class__.__name__} expect non-binary data'
        assert isinstance(self.binary_method, BaseQuantifier), f'{self.binary_method} does not seem to be a Quantifier'
        self.class_method = {c: deepcopy(self.binary_method) for c in data.classes_}
        Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(self._delayed_binary_fit)(c, self.class_method, data, **kwargs) for c in data.classes_
        )
        return self

    def quantify(self, X, *args):
        prevalences = np.asarray(
            Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(self._delayed_binary_predict)(c, self.class_method, X) for c in self.classes
            )
        )
        print('one vs all: ', prevalences)
        return F.normalize_prevalence(prevalences)

    @property
    def classes(self):
        return sorted(self.class_method.keys())

    def set_params(self, **parameters):
        self.binary_method.set_params(**parameters)

    def get_params(self, deep=True):
        return self.binary_method.get_params()

    def _delayed_binary_predict(self, c, learners, X):
        return learners[c].classify(X).mean()  # the mean is the estimation for the positive class prevalence

    def _delayed_binary_fit(self, c, learners, data, **kwargs):
        bindata = LabelledCollection(data.instances, data.labels == c, n_classes=2)
        learners[c].fit(bindata, **kwargs)


class ExplicitLossMinimisation(AggregativeQuantifier):
    """
    A variant of Explicit Loss Minimisation based on SVMperf that works also on single-label data. It uses one binary
    quantifier for each class and then l1-normalizes the class predictions so that they sum up to one.
    This variant was used in Gao, W., Sebastiani, F.: From classification to quantification in tweet sentiment analysis.
    Social Network Analysis and Mining6(19), 1–22 (2016)
    """

    def __init__(self, svmperf_base, loss, **kwargs):
        self.svmperf_base = svmperf_base
        self.loss = loss
        self.kwargs = kwargs

    def fit(self, data: LabelledCollection, fit_learner=True, *args):
        assert fit_learner, 'the method requires that fit_learner=True'
        self.learner = ExplicitLossMinimisationBinary(self.svmperf_base, self.loss, **self.kwargs)
        if not data.binary:
            self.learner = OneVsAll(self.learner, n_jobs=-1)
        return self.learner.fit(data, *args)

    def quantify(self, instances, *args):
        return self.learner.quantify(instances, *args)


class ExplicitLossMinimisationBinary(AggregativeQuantifier):

    def __init__(self, svmperf_base, loss, **kwargs):
        self.svmperf_base = svmperf_base
        self.loss = loss
        self.kwargs = kwargs

    def fit(self, data: LabelledCollection, fit_learner=True, *args):
        assert data.binary, f'{self.__class__.__name__} works only on problems of binary classification'
        assert fit_learner, 'the method requires that fit_learner=True'
        self.learner = SVMperf(self.svmperf_base, loss=self.loss, **self.kwargs).fit(data.instances, data.labels)
        return self

    def quantify(self, X, y=None):
        predictions = self.learner.predict(X)
        prev = F.prevalence_from_labels(predictions, self.learner.n_classes_)
        print('binary: ', prev)
        return prev

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

