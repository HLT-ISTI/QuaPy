from functools import cached_property
import numpy as np
import quapy as qp
import quapy.functional as F
from quapy.data import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier
from scipy.stats import chi2
from scipy.special import gamma
from sklearn.utils import resample
from abc import ABC, abstractmethod
from scipy.special import softmax, factorial
import copy
from functools import lru_cache



class ConfidenceRegionABC(ABC):

    @abstractmethod
    def point_estimate(self) -> np.ndarray:
        ...

    def ndim(self):
        return len(self.point_estimate())

    @abstractmethod
    def coverage(self, true_value):
        ...

    @lru_cache
    def simplex_portion(self):
        return self.montecarlo_proportion()

    @lru_cache
    def montecarlo_proportion(self, n_trials=10_000):
        with qp.util.temp_seed(0):
            uniform_simplex = F.uniform_simplex_sampling(n_classes=self.ndim(), size=n_trials)
        proportion = np.clip(self.coverage(uniform_simplex), 0., 1.)
        return proportion


class WithConfidenceABC(ABC):
    @abstractmethod
    def quantify_conf(self, instances, confidence_level=None) -> (np.ndarray, ConfidenceRegionABC):
        ...


def simplex_volume(n):
    return 1 / factorial(n)


def within_ellipse_prop(values, mean, prec_matrix, chi2_critical):
    """
    Checks the proportion of values that belong to the ellipse with center `mean` and precision matrix `prec_matrix`
    at a distance `chi2_critical`.

    :param values: a np.ndarray with shape (ndim,) or (n_values,ndim,)
    :param mean: a np.ndarray with the mean of the sample
    :param prec_matrix: a np.ndarray with the precision matrix (inverse of the
        covariance matrix) of the sample. If this inverse cannot be computed
        then None must be passed
    :param chi2_critical: the chi2 critical value

    :return: the fraction of values that are contained in the ellipse
        defined by the mean, the precision matrix, and the chi2_critical.
        If values is only one value, then either 0 (not contained) or
        1 (contained) is returned.
    """
    if prec_matrix is None:
        return 0.

    diff = values - mean  # Mahalanobis distance

    d_M_squared = diff @ prec_matrix @ diff.T  # d_M^2
    if d_M_squared.ndim == 2:
        d_M_squared = np.diag(d_M_squared)

    within_elipse = (d_M_squared <= chi2_critical)

    if isinstance(within_elipse, np.ndarray):
        within_elipse = np.mean(within_elipse)

    return within_elipse * 1.0


class ConfidenceEllipseSimplex(ConfidenceRegionABC):

    def __init__(self, X, confidence_level=0.95):

        assert 0. < confidence_level < 1., f'{confidence_level=} must be in range(0,1)'

        X = np.asarray(X)

        self.mean_ = X.mean(axis=0)
        self.cov_ = np.cov(X, rowvar=False, ddof=1)

        try:
            self.precision_matrix_ = np.linalg.inv(self.cov_)
        except:
            self.precision_matrix_ = None

        self.dim = X.shape[-1]
        self.ddof = self.dim - 1

        # critical chi-square value
        self.confidence_level = confidence_level
        self.chi2_critical_ = chi2.ppf(confidence_level, df=self.ddof)

    def point_estimate(self):
        return self.mean_

    def coverage(self, true_value):
        """
        true_value can be an array (n_dimensions,) or a matrix (n_vectors, n_dimensions,)
        confidence_level None means that the confidence_level is taken from the __init__
        returns true or false depending on whether true_value is in the ellipse or not,
            or returns the proportion of true_values that are within the ellipse if more
            than one are passed
        """
        return within_ellipse_prop(true_value, self.mean_, self.precision_matrix_, self.chi2_critical_)


class ConfidenceEllipseCLR(ConfidenceRegionABC):

    def __init__(self, X, confidence_level=0.95):
        self.clr = CLRtransformation()
        Z = self.clr(X)
        self.mean_ = np.mean(X, axis=0)
        self.conf_region_clr = ConfidenceEllipseSimplex(Z, confidence_level=confidence_level)

    def point_estimate(self):
        # Z_mean = self.conf_region_clr.mean()
        # return self.clr.inverse(Z_mean)
        # the inverse of the CLR does not coincide with the clean mean because the geometric mean
        # requires smoothing the prevalence vectors and this affects the softmax (inverse)
        return self.mean_

    def coverage(self, true_value):
        """
        true_value can be an array (n_dimensions,) or a matrix (n_vectors, n_dimensions,)
        confidence_level None means that the confidence_level is taken from the __init__
        returns true or false depending on whether true_value is in the ellipse or not,
            or returns the proportion of true_values that are within the ellipse if more
            than one are passed
        """
        transformed_values = self.clr(true_value)
        return self.conf_region_clr.coverage(transformed_values)


class ConfidenceIntervals(ConfidenceRegionABC):

    def __init__(self, X, confidence_level=0.95):
        assert 0 < confidence_level < 1, f'{confidence_level=} must be in range(0,1)'

        X = np.asarray(X)

        self.means_ = X.mean(axis=0)
        self.I_low, self.I_high = np.percentile(X, q=[2.5, 97.5], axis=0)

    def point_estimate(self):
        return self.means_

    def coverage(self, true_value):
        """
        true_value can be an array (n_dimensions,) or a matrix (n_vectors, n_dimensions,)
        returns true or false depending on whether true_value is in the ellipse or not,
            or returns the proportion of true_values that are within the ellipse if more
            than one are passed
        """
        within_intervals = np.logical_and(self.I_low <= true_value, true_value <= self.I_high)
        within_all_intervals = np.all(within_intervals, axis=-1, keepdims=True)
        proportion = within_all_intervals.mean()

        return proportion


class CLRtransformation:
    """
    Centered log-ratio
    """

    def __call__(self, X, epsilon=1e-6):
        X = np.asarray(X)
        X = qp.error.smooth(X, epsilon)
        G = np.exp(np.mean(np.log(X), axis=-1, keepdims=True))  # geometric mean
        return np.log(X / G)

    def inverse(self, X):
        return softmax(X, axis=-1)


class AggregativeBootstrap(WithConfidenceABC, AggregativeQuantifier):

    METHODS = ['intervals', 'ellipse', 'ellipse-clr']

    def __init__(self,
                 quantifier: AggregativeQuantifier,
                 n_train_samples=1,
                 n_test_samples=500,
                 confidence_level=0.95,
                 method='intervals',
                 random_state=None):

        assert isinstance(quantifier, AggregativeQuantifier), \
            f'base quantifier does not seem to be an instance of {AggregativeQuantifier.__name__}'
        assert n_train_samples >= 1, \
            f'{n_train_samples=} must be >= 1'
        assert n_test_samples >= 1, \
            f'{n_test_samples=} must be >= 1'
        assert n_test_samples>1 or n_train_samples>1, \
            f'either {n_test_samples=} or {n_train_samples=} must be >1'
        assert method in self.METHODS, \
            f'unknown method; valid ones are {self.METHODS}'

        self.quantifier = quantifier
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.confidence_level = confidence_level
        self.method = method
        self.random_state = random_state

    def _return_conf(self, prevs, confidence_level):
        region = None
        if self.method == 'intervals':
            region = ConfidenceIntervals(prevs, confidence_level=confidence_level)
        elif self.method == 'ellipse':
            region = ConfidenceEllipseSimplex(prevs, confidence_level=confidence_level)
        elif self.method == 'ellipse-clr':
            region = ConfidenceEllipseCLR(prevs, confidence_level=confidence_level)

        if region is None:
            raise NotImplementedError(f'unknown method {self.method}')

        return region

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        self.quantifiers = []
        if self.n_train_samples==1:
            self.quantifier.aggregation_fit(classif_predictions, data)
            self.quantifiers.append(self.quantifier)
        else:
            # model-based bootstrap (only on the aggregative part)
            full_index = np.arange(len(data))
            with qp.util.temp_seed(self.random_state):
                for i in range(self.n_train_samples):
                    quantifier = copy.deepcopy(self.quantifier)
                    index = resample(full_index, n_samples=len(data))
                    classif_predictions_i = classif_predictions.sampling_from_index(index)
                    data_i = data.sampling_from_index(index)
                    quantifier.aggregation_fit(classif_predictions_i, data_i)
                    self.quantifiers.append(quantifier)
        return self

    def aggregate(self, classif_predictions: np.ndarray):
        prev_mean, self.confidence = self.aggregate_conf(classif_predictions)
        return prev_mean

    def aggregate_conf(self, classif_predictions: np.ndarray, confidence_level=None):
        if confidence_level is None:
            confidence_level = self.confidence_level

        n_samples = classif_predictions.shape[0]
        prevs = []
        with qp.util.temp_seed(self.random_state):
            for quantifier in self.quantifiers:
                for i in range(self.n_test_samples):
                    sample_i = resample(classif_predictions, n_samples=n_samples)
                    prev_i = quantifier.aggregate(sample_i)
                    prevs.append(prev_i)

        conf = self._return_conf(prevs, confidence_level)
        prev_estim = conf.point_estimate()

        return prev_estim, conf

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split=None):
        self.quantifier._check_init_parameters()
        classif_predictions = self.quantifier.classifier_fit_predict(data, fit_classifier, predict_on=val_split)
        self.aggregation_fit(classif_predictions, data)
        return self

    def quantify_conf(self, instances, confidence_level=None) -> (np.ndarray, ConfidenceRegionABC):
        predictions = self.quantifier.classify(instances)
        return self.aggregate_conf(predictions, confidence_level=confidence_level)

    @property
    def classifier(self):
        return self.quantifier.classifier

    def _classifier_method(self):
        return self.quantifier._classifier_method()
