from typing import Union, Callable
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import resample
from sklearn.preprocessing import normalize

from method.confidence import WithConfidenceABC, ConfidenceRegionABC
from quapy.functional import get_divergence
from quapy.method.base import BaseQuantifier, BinaryQuantifier
import quapy.functional as F
from scipy.optimize import lsq_linear


class MaximumLikelihoodPrevalenceEstimation(BaseQuantifier):
    """
    The `Maximum Likelihood Prevalence Estimation` (MLPE) method is a lazy method that assumes there is no prior
    probability shift between training and test instances (put it other way, that the i.i.d. assumpion holds).
    The estimation of class prevalence values for any test sample is always (i.e., irrespective of the test sample
    itself) the class prevalence seen during training. This method is considered to be a lower-bound quantifier that
    any quantification method should beat.
    """

    def __init__(self):
        self._classes_ = None

    def fit(self, X, y):
        """
        Computes the training prevalence and stores it.

        :param X: array-like of shape `(n_samples, n_features)`, the training instances
        :param y: array-like of shape `(n_samples,)`, the labels
        :return: self
        """
        self._classes_ = F.classes_from_labels(labels=y)
        self.estimated_prevalence = F.prevalence_from_labels(y, classes=self._classes_)
        return self

    def predict(self, X):
        """
        Ignores the input instances and returns, as the class prevalence estimantes, the training prevalence.

        :param X: array-like (ignored)
        :return: the class prevalence seen during training
        """
        return self.estimated_prevalence


class DMx(BaseQuantifier):
    """
    Generic Distribution Matching quantifier for binary or multiclass quantification based on the space of covariates.
    This implementation takes the number of bins, the divergence, and the possibility to work on CDF as hyperparameters.

    :param nbins: number of bins used to discretize the distributions (default 8)
    :param divergence: a string representing a divergence measure (currently, "HD" and "topsoe" are implemented)
        or a callable function taking two ndarrays of the same dimension as input (default "HD", meaning Hellinger
        Distance)
    :param cdf: whether to use CDF instead of PDF (default False)
    :param n_jobs: number of parallel workers (default None)
    """

    def __init__(self, nbins=8, divergence: Union[str, Callable]='HD', cdf=False, search='optim_minimize', n_jobs=None):
        self.nbins = nbins
        self.divergence = divergence
        self.cdf = cdf
        self.search = search
        self.n_jobs = n_jobs

    @classmethod
    def HDx(cls, n_jobs=None):
        """
        `Hellinger Distance x <https://www.sciencedirect.com/science/article/pii/S0020025512004069>`_ (HDx).
        HDx is a method for training binary quantifiers, that models quantification as the problem of
        minimizing the average divergence (in terms of the Hellinger Distance) across the feature-specific normalized
        histograms of two representations, one for the unlabelled examples, and another generated from the training
        examples as a mixture model of the class-specific representations. The parameters of the mixture thus represent
        the estimates of the class prevalence values.

        The method computes all matchings for nbins in [10, 20, ..., 110] and reports the mean of the median.
        The best prevalence is searched via linear search, from 0 to 1 stepping by 0.01.

        :param n_jobs: number of parallel workers
        :return: an instance of this class setup to mimick the performance of the HDx as originally proposed by
            González-Castro, Alaiz-Rodríguez, Alegre (2013)
        """
        from quapy.method.meta import MedianEstimator

        dmx = DMx(divergence='HD', cdf=False, search='linear_search')
        nbins = {'nbins': np.linspace(10, 110, 11, dtype=int)}
        hdx = MedianEstimator(base_quantifier=dmx, param_grid=nbins, n_jobs=n_jobs)
        return hdx

    def __get_distributions(self, X):

        histograms = []
        for feat_idx in range(self.nfeats):
            feature = X[:, feat_idx]
            feat_range = self.feat_ranges[feat_idx]
            hist = np.histogram(feature, bins=self.nbins, range=feat_range)[0]
            norm_hist = hist / hist.sum()
            histograms.append(norm_hist)
        distributions = np.vstack(histograms)

        if self.cdf:
            distributions = np.cumsum(distributions, axis=1)

        return distributions

    def fit(self, X, y):
        """
        Generates the validation distributions out of the training data (covariates).
        The validation distributions have shape `(n, nfeats, nbins)`, with `n` the number of classes, `nfeats`
        the number of features, and `nbins` the number of bins.
        In particular, let `V` be the validation distributions; then `di=V[i]` are the distributions obtained from
        training data labelled with class `i`; while `dij = di[j]` is the discrete distribution for feature j in
        training data labelled with class `i`, and `dij[k]` is the fraction of instances with a value in the `k`-th bin.

        :param X: array-like of shape `(n_samples, n_features)`, the training instances
        :param y: array-like of shape `(n_samples,)`, the labels
        """
        self.nfeats = X.shape[1]
        self.feat_ranges = _get_features_range(X)
        n_classes = len(np.unique(y))

        self.validation_distribution = np.asarray(
            [self.__get_distributions(X[y==cat]) for cat in range(n_classes)]
        )

        return self

    def predict(self, X):
        """
        Searches for the mixture model parameter (the sought prevalence values) that yields a validation distribution
        (the mixture) that best matches the test distribution, in terms of the divergence measure of choice.
        The matching is computed as the average dissimilarity (in terms of the dissimilarity measure of choice)
        between all feature-specific discrete distributions.

        :param X: instances in the sample
        :return: a vector of class prevalence estimates
        """

        assert X.shape[1] == self.nfeats, f'wrong shape; expected {self.nfeats}, found {X.shape[1]}'

        test_distribution = self.__get_distributions(X)
        divergence = get_divergence(self.divergence)
        n_classes, n_feats, nbins = self.validation_distribution.shape
        def loss(prev):
            prev = np.expand_dims(prev, axis=0)
            mixture_distribution = (prev @ self.validation_distribution.reshape(n_classes,-1)).reshape(n_feats, -1)
            divs = [divergence(test_distribution[feat], mixture_distribution[feat]) for feat in range(n_feats)]
            return np.mean(divs)

        return F.argmin_prevalence(loss, n_classes, method=self.search)


class ReadMe(BaseQuantifier, WithConfidenceABC):

    def __init__(self,
                 bootstrap_trials=100,
                 bagging_trials=100,
                 bagging_range=250,
                 confidence_level=0.95,
                 region='intervals',
                 random_state=None,
                 verbose=False):
        self.bootstrap_trials = bootstrap_trials
        self.bagging_trials = bagging_trials
        self.bagging_range = bagging_range
        self.confidence_level = confidence_level
        self.region = region
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        self.rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        n_features = X.shape[1]

        if self.bagging_range is None:
            self.bagging_range = int(np.sqrt(n_features))

        Xsize = X.shape[0]

        # Bootstrap loop
        self.Xboots, self.yboots = [], []
        for _ in range(self.bootstrap_trials):
            idx = self.rng.choice(Xsize, size=Xsize, replace=True)
            self.Xboots.append(X[idx])
            self.yboots.append(y[idx])

        return self

    def predict_conf(self, X, confidence_level=0.95) -> (np.ndarray, ConfidenceRegionABC):
        from tqdm import tqdm
        n_features = X.shape[1]

        boots_prevalences = []

        for Xboots, yboots in tqdm(
                zip(self.Xboots, self.yboots),
                desc='bootstrap predictions', total=self.bootstrap_trials, disable=not self.verbose
        ):
            bagging_estimates = []
            for _ in range(self.bagging_trials):
                feat_idx = self.rng.choice(n_features, size=self.bagging_range, replace=False)
                Xboots_bagging = Xboots[:, feat_idx]
                X_boots_bagging = X[:, feat_idx]
                bagging_prev = self._quantify_iteration(Xboots_bagging, yboots, X_boots_bagging)
                bagging_estimates.append(bagging_prev)

            boots_prevalences.append(np.mean(bagging_estimates, axis=0))

        conf = WithConfidenceABC.construct_region(boots_prevalences, confidence_level, method=self.region)
        prev_estim = conf.point_estimate()

        return prev_estim, conf


    def predict(self, X):
        prev_estim, _ = self.predict_conf(X)
        return prev_estim


    def _quantify_iteration(self, Xtr, ytr, Xte):
        """Single ReadMe estimate."""
        n_classes = len(self.classes_)
        PX_given_Y = np.zeros((n_classes, Xtr.shape[1]))
        for i, c in enumerate(self.classes_):
            PX_given_Y[i] = Xtr[ytr == c].sum(axis=0)
        PX_given_Y = normalize(PX_given_Y, norm='l1', axis=1)

        PX = np.asarray(Xte.sum(axis=0))
        PX = normalize(PX, norm='l1', axis=1)

        res = lsq_linear(A=PX_given_Y.T, b=PX.ravel(), bounds=(0, 1))
        pY = np.maximum(res.x, 0)
        return pY / pY.sum()



def _get_features_range(X):
    feat_ranges = []
    ncols = X.shape[1]
    for col_idx in range(ncols):
        feature = X[:,col_idx]
        feat_ranges.append((np.min(feature), np.max(feature)))
    return feat_ranges


#---------------------------------------------------------------
# aliases
#---------------------------------------------------------------

DistributionMatchingX = DMx