from typing import Union, Callable

import numpy as np
from scipy import optimize

from functional import get_divergence
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



class HDx(BinaryQuantifier):
    """
    `Hellinger Distance x <https://www.sciencedirect.com/science/article/pii/S0020025512004069>`_ (HDx).
    HDx is a method for training binary quantifiers, that models quantification as the problem of
    minimizing the average divergence (in terms of the Hellinger Distance) across the feature-specific normalized
    histograms of two representations, one for the unlabelled examples, and another generated from the training
    examples as a mixture model of the class-specific representations. The parameters of the mixture thus represent
    the estimates of the class prevalence values. The method computes all matchings for nbins in [10, 20, ..., 110]
    and reports the mean of the median. The best prevalence is searched via linear search, from 0 to 1 steppy by 0.01.
    """

    def __init__(self):
        self.feat_ranges = None

    def covariate_histograms(self, X, nbins):
        assert self.feat_ranges is not None, 'quantify called before fit'

        histograms = []
        for col_idx in range(self.nfeats):
            feature = X[:,col_idx]
            feat_range = self.feat_ranges[col_idx]
            histograms.append(np.histogram(feature, bins=nbins, range=feat_range, density=True)[0])

        return np.vstack(histograms).T

    def fit(self, data: LabelledCollection):
        """
        Trains a HDx quantifier.

        :param data: the training set
        :return: self
        """

        self._check_binary(data, self.__class__.__name__)
        X, y = data.Xy

        self.nfeats = X.shape[1]
        self.feat_ranges = _get_features_range(X)

        # pre-compute the representation for positive and negative examples
        self.bins = np.linspace(10, 110, 11, dtype=int)  # [10, 20, 30, ..., 100, 110]
        self.H0 = {bins:self.covariate_histograms(X[y == 0], bins) for bins in self.bins}
        self.H1 = {bins:self.covariate_histograms(X[y == 1], bins) for bins in self.bins}
        return self

    def quantify(self, X):
        # "In this work, the number of bins b used in HDx and HDy was chosen from 10 to 110 in steps of 10,
        # and the final estimated a priori probability was taken as the median of these 11 estimates."
        # (González-Castro, et al., 2013).

        assert X.shape[1] == self.nfeats, f'wrong shape in quantify; expected {self.nfeats}, found {X.shape[1]}'

        prev_estimations = []
        for nbins in self.bins:
            Ht = self.covariate_histograms(X, nbins=nbins)
            H0 = self.H0[nbins]
            H1 = self.H1[nbins]

            # the authors proposed to search for the prevalence yielding the best matching as a linear search
            # at small steps (modern implementations resort to an optimization procedure)
            prev_selected, min_dist = None, None
            for prev in F.prevalence_linspace(n_prevalences=100, repeats=1, smooth_limits_epsilon=0.0):
                Hx = prev * H1 + (1 - prev) * H0
                hdx = np.mean([F.HellingerDistance(Hx[:,col], Ht[:,col]) for col in range(self.nfeats)])

                if prev_selected is None or hdx < min_dist:
                    prev_selected, min_dist = prev, hdx
            prev_estimations.append(prev_selected)

        class1_prev = np.median(prev_estimations)
        return np.asarray([1 - class1_prev, class1_prev])


class DistributionMatchingX(BaseQuantifier):
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

    def __init__(self, nbins=8, divergence: Union[str, Callable]='HD', cdf=False, n_jobs=None):
        self.nbins = nbins
        self.divergence = divergence
        self.cdf = cdf
        self.n_jobs = n_jobs

    def __get_distributions(self, X):
        histograms = []
        for feat_idx in range(self.nfeats):
            hist = np.histogram(X[:, feat_idx], bins=self.nbins, density=True, range=self.feat_ranges[feat_idx])[0]
            histograms.append(hist)

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
        def match(prev):
            prev = np.expand_dims(prev, axis=0)
            mixture_distribution = (prev @ self.validation_distribution.reshape(n_classes,-1)).reshape(n_feats, -1)
            divs = [divergence(test_distribution[feat], mixture_distribution[feat]) for feat in range(n_feats)]
            return np.mean(divs)

        # the initial point is set as the uniform distribution
        uniform_distribution = np.full(fill_value=1 / n_classes, shape=(n_classes,))

        # solutions are bounded to those contained in the unit-simplex
        bounds = tuple((0, 1) for x in range(n_classes))  # values in [0,1]
        constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})  # values summing up to 1
        r = optimize.minimize(match, x0=uniform_distribution, method='SLSQP', bounds=bounds, constraints=constraints)
        return r.x


def _get_features_range(X):
    feat_ranges = []
    ncols = X.shape[1]
    for col_idx in range(ncols):
        feature = X[:,col_idx]
        feat_ranges.append((np.min(feature), np.max(feature)))
    return feat_ranges