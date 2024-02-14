from typing import Union, Callable
import numpy as np

from quapy.functional import get_divergence
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier, BinaryQuantifier
import quapy.functional as F


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

    def fit(self, data: LabelledCollection):
        """
        Computes the training prevalence and stores it.

        :param data: the training sample
        :return: self
        """
        self.estimated_prevalence = data.prevalence()
        return self

    def quantify(self, instances):
        """
        Ignores the input instances and returns, as the class prevalence estimantes, the training prevalence.

        :param instances: array-like (ignored)
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

    def fit(self, data: LabelledCollection):
        """
        Generates the validation distributions out of the training data (covariates).
        The validation distributions have shape `(n, nfeats, nbins)`, with `n` the number of classes, `nfeats`
        the number of features, and `nbins` the number of bins.
        In particular, let `V` be the validation distributions; then `di=V[i]` are the distributions obtained from
        training data labelled with class `i`; while `dij = di[j]` is the discrete distribution for feature j in
        training data labelled with class `i`, and `dij[k]` is the fraction of instances with a value in the `k`-th bin.

        :param data: the training set
        """
        X, y = data.Xy

        self.nfeats = X.shape[1]
        self.feat_ranges = _get_features_range(X)

        self.validation_distribution = np.asarray(
            [self.__get_distributions(X[y==cat]) for cat in range(data.n_classes)]
        )

        return self

    def quantify(self, instances):
        """
        Searches for the mixture model parameter (the sought prevalence values) that yields a validation distribution
        (the mixture) that best matches the test distribution, in terms of the divergence measure of choice.
        The matching is computed as the average dissimilarity (in terms of the dissimilarity measure of choice)
        between all feature-specific discrete distributions.

        :param instances: instances in the sample
        :return: a vector of class prevalence estimates
        """

        assert instances.shape[1] == self.nfeats, f'wrong shape; expected {self.nfeats}, found {instances.shape[1]}'

        test_distribution = self.__get_distributions(instances)
        divergence = get_divergence(self.divergence)
        n_classes, n_feats, nbins = self.validation_distribution.shape
        def loss(prev):
            prev = np.expand_dims(prev, axis=0)
            mixture_distribution = (prev @ self.validation_distribution.reshape(n_classes,-1)).reshape(n_feats, -1)
            divs = [divergence(test_distribution[feat], mixture_distribution[feat]) for feat in range(n_feats)]
            return np.mean(divs)

        return F.argmin_prevalence(loss, n_classes, method=self.search)



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