import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

import quapy as qp
import quapy.functional as F
from quapy.method import _bayesian
from quapy.method.aggregative import AggregativeCrispQuantifier
from quapy.data import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier
from scipy.stats import chi2
from sklearn.utils import resample
from abc import ABC, abstractmethod
from scipy.special import softmax, factorial
import copy
from functools import lru_cache

"""
This module provides implementation of different types of confidence regions, and the implementation of Bootstrap
for AggregativeQuantifiers.
"""

class ConfidenceRegionABC(ABC):
    """
    Abstract class of confidence regions
    """

    @abstractmethod
    def point_estimate(self) -> np.ndarray:
        """
        Returns the point estimate corresponding to a set of bootstrap estimates.

        :return: np.ndarray
        """
        ...

    def ndim(self) -> int:
        """
        Number of dimensions of the region. This number corresponds to the total number of classes. The dimensionality
        of the simplex is therefore ndim-1

        :return: int
        """
        return len(self.point_estimate())

    @abstractmethod
    def coverage(self, true_value) -> float:
        """
        Checks whether a value, or a sets of values, are contained in the confidence region. The method computes the
        fraction of these that are contained in the region, if more than one value is passed. If only one value is
        passed, then it either returns 1.0 or 0.0, for indicating the value is in the region or not, respectively.

        :param true_value: a np.ndarray of shape (n_classes,) or shape (n_values, n_classes,)
        :return: float in [0,1]
        """
        ...

    @lru_cache
    def simplex_portion(self):
        """
        Computes the fraction of the simplex which is covered by the region. This is not the volume of the region
        itself (which could lie outside the boundaries of the simplex), but the actual fraction of the simplex
        contained in the region. A default implementation, based on Monte Carlo approximation, is provided.

        :return: float, the fraction of the simplex covered by the region
        """
        return self.montecarlo_proportion()

    @lru_cache
    def montecarlo_proportion(self, n_trials=10_000):
        """
        Estimates, via a Monte Carlo approach, the fraction of the simplex covered by the region. This is carried
        out by returning the fraction of the `n_trials` points, uniformly drawn at random from the simplex, that
        are included in the region. The value is only computed once when multiple calls are made.

        :return: float in [0,1]
        """
        with qp.util.temp_seed(0):
            uniform_simplex = F.uniform_simplex_sampling(n_classes=self.ndim(), size=n_trials)
        proportion = np.clip(self.coverage(uniform_simplex), 0., 1.)
        return proportion


class WithConfidenceABC(ABC):
    """
    Abstract class for confidence regions.
    """
    METHODS = ['intervals', 'ellipse', 'ellipse-clr']

    @abstractmethod
    def quantify_conf(self, instances, confidence_level=None) -> (np.ndarray, ConfidenceRegionABC):
        """
        Adds the method `quantify_conf` to the interface. This method returns not only the point-estimate, but
        also the confidence region around it.

        :param instances: a np.ndarray of shape (n_instances, n_features,)
        :confidence_level: float in (0, 1)
        :return: a tuple (`point_estimate`, `conf_region`), where `point_estimate` is a np.ndarray of shape
            (n_classes,) and  `conf_region` is an object from :class:`ConfidenceRegionABC`
        """
        ...

    @classmethod
    def construct_region(cls, prev_estims, confidence_level=0.95, method='intervals'):
        """
        Construct a confidence region given many prevalence estimations.

        :param prev_estims: np.ndarray of shape (n_estims, n_classes)
        :param confidence_level: float, the confidence level for the region (default 0.95)
        :param method: str, indicates the method for constructing regions. Set to `intervals` for
            constructing confidence intervals (default), or to `ellipse` for constructing an
            ellipse in the probability simplex, or to `ellipse-clr` for constructing an ellipse
            in the Centered-Log Ratio (CLR) unconstrained space.
        """
        region = None
        if method == 'intervals':
            region = ConfidenceIntervals(prev_estims, confidence_level=confidence_level)
        elif method == 'ellipse':
            region = ConfidenceEllipseSimplex(prev_estims, confidence_level=confidence_level)
        elif method == 'ellipse-clr':
            region = ConfidenceEllipseCLR(prev_estims, confidence_level=confidence_level)

        if region is None:
            raise NotImplementedError(f'unknown method {method}')

        return region

def simplex_volume(n):
    """
    Computes the volume of the n-dimensional simplex. For n classes, the corresponding volume
    is :meth:`simplex_volume(n-1)` since the simplex has one degree of freedom less.

    :param n: int, the dimensionality of the simplex
    :return: float, the volume of the n-dimensional simplex
    """
    return 1 / factorial(n)


def within_ellipse_prop(values, mean, prec_matrix, chi2_critical):
    """
    Checks the proportion of values that belong to the ellipse with center `mean` and precision matrix `prec_matrix`
    at a distance `chi2_critical`.

    :param values: a np.ndarray of shape (n_dim,) or (n_values, n_dim,)
    :param mean: a np.ndarray of shape (n_dim,) with the center of the ellipse
    :param prec_matrix: a np.ndarray with the precision matrix (inverse of the
        covariance matrix) of the ellipse. If this inverse cannot be computed
        then None must be passed
    :param chi2_critical: float, the chi2 critical value

    :return: float in [0,1], the fraction of values that are contained in the ellipse
        defined by the mean (center), the precision matrix (shape), and the chi2_critical value (distance).
        If `values` is only one value, then either 0. (not contained) or 1. (contained) is returned.
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
    """
    Instantiates a Confidence Ellipse in the probability simplex.

    :param X: np.ndarray of shape (n_bootstrap_samples, n_classes)
    :param confidence_level: float, the confidence level (default 0.95)
    """

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
        """
        Returns the point estimate, the center of the ellipse.

        :return: np.ndarray of shape (n_classes,)
        """
        return self.mean_

    def coverage(self, true_value):
        """
        Checks whether a value, or a sets of values, are contained in the confidence region. The method computes the
        fraction of these that are contained in the region, if more than one value is passed. If only one value is
        passed, then it either returns 1.0 or 0.0, for indicating the value is in the region or not, respectively.

        :param true_value: a np.ndarray of shape (n_classes,) or shape (n_values, n_classes,)
        :return: float in [0,1]
        """
        return within_ellipse_prop(true_value, self.mean_, self.precision_matrix_, self.chi2_critical_)


class ConfidenceEllipseCLR(ConfidenceRegionABC):
    """
    Instantiates a Confidence Ellipse in the Centered-Log Ratio (CLR) space.

    :param X: np.ndarray of shape (n_bootstrap_samples, n_classes)
    :param confidence_level: float, the confidence level (default 0.95)
    """

    def __init__(self, X, confidence_level=0.95):
        self.clr = CLRtransformation()
        Z = self.clr(X)
        self.mean_ = np.mean(X, axis=0)
        self.conf_region_clr = ConfidenceEllipseSimplex(Z, confidence_level=confidence_level)

    def point_estimate(self):
        """
        Returns the point estimate, the center of the ellipse.

        :return: np.ndarray of shape (n_classes,)
        """
        # The inverse of the CLR does not coincide with the true mean, because the geometric mean
        # requires smoothing the prevalence vectors and this affects the softmax (inverse);
        # return self.clr.inverse(self.mean_) # <- does not coincide
        return self.mean_

    def coverage(self, true_value):
        """
        Checks whether a value, or a sets of values, are contained in the confidence region. The method computes the
        fraction of these that are contained in the region, if more than one value is passed. If only one value is
        passed, then it either returns 1.0 or 0.0, for indicating the value is in the region or not, respectively.

        :param true_value: a np.ndarray of shape (n_classes,) or shape (n_values, n_classes,)
        :return: float in [0,1]
        """
        transformed_values = self.clr(true_value)
        return self.conf_region_clr.coverage(transformed_values)


class ConfidenceIntervals(ConfidenceRegionABC):
    """
    Instantiates a region based on (independent) Confidence Intervals.

    :param X: np.ndarray of shape (n_bootstrap_samples, n_classes)
    :param confidence_level: float, the confidence level (default 0.95)
    """
    def __init__(self, X, confidence_level=0.95):
        assert 0 < confidence_level < 1, f'{confidence_level=} must be in range(0,1)'

        X = np.asarray(X)

        self.means_ = X.mean(axis=0)
        alpha = 1-confidence_level
        low_perc = (alpha/2.)*100
        high_perc = (1-alpha/2.)*100
        self.I_low, self.I_high = np.percentile(X, q=[low_perc, high_perc], axis=0)

    def point_estimate(self):
        """
        Returns the point estimate, the class-wise average of the bootstrapped estimates

        :return: np.ndarray of shape (n_classes,)
        """
        return self.means_

    def coverage(self, true_value):
        """
        Checks whether a value, or a sets of values, are contained in the confidence region. The method computes the
        fraction of these that are contained in the region, if more than one value is passed. If only one value is
        passed, then it either returns 1.0 or 0.0, for indicating the value is in the region or not, respectively.

        :param true_value: a np.ndarray of shape (n_classes,) or shape (n_values, n_classes,)
        :return: float in [0,1]
        """
        within_intervals = np.logical_and(self.I_low <= true_value, true_value <= self.I_high)
        within_all_intervals = np.all(within_intervals, axis=-1, keepdims=True)
        proportion = within_all_intervals.mean()

        return proportion


class CLRtransformation:
    """
    Centered log-ratio, from component analysis
    """
    def __call__(self, X, epsilon=1e-6):
        """
        Applies the CLR function to X thus mapping the instances, which are contained in `\\mathcal{R}^{n}` but
        actually lie on a `\\mathcal{R}^{n-1}` simplex, onto an unrestricted space in :math:`\\mathcal{R}^{n}`

        :param X: np.ndarray of (n_instances, n_dimensions) to be transformed
        :param epsilon: small float for prevalence smoothing
        :return: np.ndarray of (n_instances, n_dimensions), the CLR-transformed points
        """
        X = np.asarray(X)
        X = qp.error.smooth(X, epsilon)
        G = np.exp(np.mean(np.log(X), axis=-1, keepdims=True))  # geometric mean
        return np.log(X / G)

    def inverse(self, X):
        """
        Inverse function. However, clr.inverse(clr(X)) does not exactly coincide with X due to smoothing.

        :param X: np.ndarray of (n_instances, n_dimensions) to be transformed
        :return: np.ndarray of (n_instances, n_dimensions), the CLR-transformed points
        """
        return softmax(X, axis=-1)


class AggregativeBootstrap(WithConfidenceABC, AggregativeQuantifier):
    """
    Aggregative Bootstrap allows any AggregativeQuantifier to get confidence regions around
    point-estimates of class prevalence values. This method implements some optimizations for
    speeding up the computations, which are only possible due to the two phases of the aggregative
    quantifiers.

    During training, the bootstrap repetitions are only carried out over pre-classified training instances,
    after the classifier has been trained (only once), in order to train a series of aggregation
    functions (model-based approach).

    During inference, the bootstrap repetitions are applied to the pre-classified test instances.

    :param quantifier: an aggregative quantifier
    :para n_train_samples: int, the number of training resamplings (defaults to 1, set to > 1 to activate a
        model-based bootstrap approach)
    :para n_test_samples: int, the number of test resamplings (defaults to 500, set to > 1 to activate a
        population-based bootstrap approach)
    :param confidence_level: float, the confidence level for the confidence region (default 0.95)
    :param region: string, set to `intervals` for constructing confidence intervals (default), or to
        `ellipse` for constructing an ellipse in the probability simplex, or to `ellipse-clr` for
        constructing an ellipse in the Centered-Log Ratio (CLR) unconstrained space.
    :param random_state: int for replicating samples, None (default) for non-replicable samples
    """

    def __init__(self,
                 quantifier: AggregativeQuantifier,
                 n_train_samples=1,
                 n_test_samples=500,
                 confidence_level=0.95,
                 region='intervals',
                 random_state=None):

        assert isinstance(quantifier, AggregativeQuantifier), \
            f'base quantifier does not seem to be an instance of {AggregativeQuantifier.__name__}'
        assert n_train_samples >= 1, \
            f'{n_train_samples=} must be >= 1'
        assert n_test_samples >= 1, \
            f'{n_test_samples=} must be >= 1'
        assert n_test_samples>1 or n_train_samples>1, \
            f'either {n_test_samples=} or {n_train_samples=} must be >1'

        self.quantifier = quantifier
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.confidence_level = confidence_level
        self.region = region
        self.random_state = random_state

    def aggregation_fit(self, classif_predictions, labels):
        data = LabelledCollection(classif_predictions, labels, classes=self.classes_)
        self.quantifiers = []
        if self.n_train_samples==1:
            self.quantifier.aggregation_fit(classif_predictions, labels)
            self.quantifiers.append(self.quantifier)
        else:
            # model-based bootstrap (only on the aggregative part)
            n_examples = len(data)
            full_index = np.arange(n_examples)
            with qp.util.temp_seed(self.random_state):
                for i in range(self.n_train_samples):
                    quantifier = copy.deepcopy(self.quantifier)
                    index = resample(full_index, n_samples=n_examples)
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

        conf = WithConfidenceABC.construct_region(prevs, confidence_level, method=self.region)
        prev_estim = conf.point_estimate()

        return prev_estim, conf

    def fit(self, X, y):
        self.quantifier._check_init_parameters()
        classif_predictions, labels = self.quantifier.classifier_fit_predict(X, y)
        self.aggregation_fit(classif_predictions, labels)
        return self

    def quantify_conf(self, instances, confidence_level=None) -> (np.ndarray, ConfidenceRegionABC):
        predictions = self.quantifier.classify(instances)
        return self.aggregate_conf(predictions, confidence_level=confidence_level)

    @property
    def classifier(self):
        return self.quantifier.classifier

    def _classifier_method(self):
        return self.quantifier._classifier_method()


class BayesianCC(AggregativeCrispQuantifier, WithConfidenceABC):
    """
    `Bayesian quantification <https://arxiv.org/abs/2302.09159>`_ method,
    which is a variant of :class:`ACC` that calculates the posterior probability distribution
    over the prevalence vectors, rather than providing a point estimate obtained
    by matrix inversion.

    Can be used to diagnose degeneracy in the predictions visible when the confusion
    matrix has high condition number or to quantify uncertainty around the point estimate.

    This method relies on extra dependencies, which have to be installed via:
    `$ pip install quapy[bayes]`

    :param classifier: a scikit-learn's BaseEstimator, or None, in which case the classifier is taken to be
        the one indicated in `qp.environ['DEFAULT_CLS']`
    :param val_split:  specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer (default 5), indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`); or as a tuple `(X,y)` defining the specific set of data to use for validation. Set to
        None when the method does not require any validation data, in order to avoid that some portion of
        the training data be wasted.
    :param num_warmup: number of warmup iterations for the MCMC sampler (default 500)
    :param num_samples: number of samples to draw from the posterior (default 1000)
    :param mcmc_seed: random seed for the MCMC sampler (default 0)
    :param confidence_level: float in [0,1] to construct a confidence region around the point estimate (default 0.95)
    :param region: string, set to `intervals` for constructing confidence intervals (default), or to
        `ellipse` for constructing an ellipse in the probability simplex, or to `ellipse-clr` for
        constructing an ellipse in the Centered-Log Ratio (CLR) unconstrained space.
    """
    def __init__(self,
                 classifier: BaseEstimator=None,
                 fit_classifier=True,
                 val_split: int = 5,
                 num_warmup: int = 500,
                 num_samples: int = 1_000,
                 mcmc_seed: int = 0,
                 confidence_level: float = 0.95,
                 region: str = 'intervals'):

        if num_warmup <= 0:
            raise ValueError(f'parameter {num_warmup=} must be a positive integer')
        if num_samples <= 0:
            raise ValueError(f'parameter {num_samples=} must be a positive integer')

        if _bayesian.DEPENDENCIES_INSTALLED is False:
            raise ImportError("Auxiliary dependencies are required. "
                              "Run `$ pip install quapy[bayes]` to install them.")

        super().__init__(classifier, fit_classifier, val_split)
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.mcmc_seed = mcmc_seed
        self.confidence_level = confidence_level
        self.region = region

        # Array of shape (n_classes, n_predicted_classes,) where entry (y, c) is the number of instances
        # labeled as class y and predicted as class c.
        # By default, this array is set to None and later defined as part of the `aggregation_fit` phase
        self._n_and_c_labeled = None

        # Dictionary with posterior samples, set when `aggregate` is provided.
        self._samples = None

    def aggregation_fit(self, classif_predictions, labels):
        """
        Estimates the misclassification rates.

        :param classif_predictions: array-like with the label predictions returned by the classifier
        :param labels: array-like with the true labels associated to each classifier prediction
        """
        pred_labels = classif_predictions
        true_labels = labels
        self._n_and_c_labeled = confusion_matrix(
            y_true=true_labels,
            y_pred=pred_labels,
            labels=self.classifier.classes_
        ).astype(float)

    def sample_from_posterior(self, classif_predictions):
        if self._n_and_c_labeled is None:
            raise ValueError("aggregation_fit must be called before sample_from_posterior")

        n_c_unlabeled = F.counts_from_labels(classif_predictions, self.classifier.classes_).astype(float)

        self._samples = _bayesian.sample_posterior(
            n_c_unlabeled=n_c_unlabeled,
            n_y_and_c_labeled=self._n_and_c_labeled,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            seed=self.mcmc_seed,
        )
        return self._samples

    def get_prevalence_samples(self):
        if self._samples is None:
            raise ValueError("sample_from_posterior must be called before get_prevalence_samples")
        return self._samples[_bayesian.P_TEST_Y]

    def get_conditional_probability_samples(self):
        if self._samples is None:
            raise ValueError("sample_from_posterior must be called before get_conditional_probability_samples")
        return self._samples[_bayesian.P_C_COND_Y]

    def aggregate(self, classif_predictions):
        samples = self.sample_from_posterior(classif_predictions)[_bayesian.P_TEST_Y]
        return np.asarray(samples.mean(axis=0), dtype=float)

    def quantify_conf(self, instances, confidence_level=None) -> (np.ndarray, ConfidenceRegionABC):
        classif_predictions = self.classify(instances)
        point_estimate = self.aggregate(classif_predictions)
        samples = self.get_prevalence_samples()  # available after calling "aggregate" function
        region = WithConfidenceABC.construct_region(samples, confidence_level=self.confidence_level, method=self.region)
        return point_estimate, region
