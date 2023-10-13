from cgi import test
import os
import sys
from typing import Union, Callable
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal
import quapy as qp
from quapy.data import LabelledCollection
from quapy.protocol import APP, UPP
from quapy.method.aggregative import AggregativeProbabilisticQuantifier, _training_helper, cross_generate_predictions, \
    DistributionMatching, _get_divergence
import scipy
from scipy import optimize
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional
from time import time
from sklearn.metrics.pairwise import rbf_kernel


def gram_matrix_mix_sum(bandwidth, X, Y=None, reduce=True):
    # this adapts the output of the rbf_kernel function (pairwise evaluations of Gaussian kernels k(x,y))
    # to contain pairwise evaluations of N(x|mu,Sigma1+Sigma2) with mu=y and Sigma1 and Sigma2 are 
    # two "scalar matrices" (h^2) I each, so Sigma1+Sigma2 has scalar 2(h^2) (h is the bandwidth)
    variance = 2 * (bandwidth**2)
    nRows,nD = X.shape
    gamma = 1/(2*variance)
    norm_factor = 1/np.sqrt(((2*np.pi)**nD) * (variance**(nD)))
    gram = norm_factor * rbf_kernel(X, Y, gamma=gamma)
    return gram.sum()


class KDEyclosed_efficient_corr(AggregativeProbabilisticQuantifier):

    def __init__(self, classifier: BaseEstimator, val_split=0.4, bandwidth=0.1, n_jobs=None, random_state=0):
        self.classifier = classifier
        self.val_split = val_split
        self.bandwidth = bandwidth
        self.n_jobs = n_jobs
        self.random_state=random_state

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split: Union[float, LabelledCollection] = None):
        """

        :param data: the training set
        :param fit_classifier: set to False to bypass the training (the learner is assumed to be already fit)
        :param val_split: either a float in (0,1) indicating the proportion of training instances to use for
         validation (e.g., 0.3 for using 30% of the training set as validation data), or a LabelledCollection
         indicating the validation set itself, or an int indicating the number k of folds to be used in kFCV
         to estimate the parameters
        """

        # print('[fit] enter')
        if val_split is None:
            val_split = self.val_split

        self.classifier, y, posteriors, classes, class_count = cross_generate_predictions(
            data, self.classifier, val_split, probabilistic=True, fit_classifier=fit_classifier, n_jobs=self.n_jobs
        )

        assert all(sorted(np.unique(y)) == np.arange(data.n_classes)), \
            'label name gaps not allowed in current implementation'

        # print('[fit] precomputing stuff')

        # from distribution_matching.binary_debug import HACK
        # posteriors, y = HACK(posteriors, y)

        n = data.n_classes
        h = self.bandwidth

        # print('[fit] classifier ready; precomputing gram')
        P = posteriors

        # li_inv keeps track of the relative weight of each datapoint within its class
        # (i.e., the weight in its KDE model)
        counts_inv = 1 / (data.counts())

        tr_tr_sums = np.zeros(shape=(n,n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i > j:
                    tr_tr_sums[i,j] = tr_tr_sums[j,i]
                else:
                    block = gram_matrix_mix_sum(h, P[y == i], P[y == j] if i!=j else None)
                    tr_tr_sums[i, j] = block

        # compute class-class block-sums
        self.Ptr = posteriors
        self.ytr = y
        self.tr_tr_sums = tr_tr_sums
        self.counts_inv = counts_inv

        return self


    def aggregate(self, posteriors: np.ndarray):

        Ptr = self.Ptr
        Pte = posteriors

        K,nD = Pte.shape
        Kinv = (1/K)
        h = self.bandwidth
        n = Ptr.shape[1]
        y = self.ytr
        tr_tr_sums = self.tr_tr_sums

        partC = 0.5*np.log(gram_matrix_mix_sum(h, Pte) * Kinv * Kinv)

        tr_te_sums = np.zeros(shape=n, dtype=float)
        for i in range(n):
            tr_te_sums[i] = gram_matrix_mix_sum(h, Ptr[y==i], Pte) * self.counts_inv[i] * Kinv

        def match(alpha):
            partA = -np.log((alpha * tr_te_sums).sum())
            alpha_l = alpha * self.counts_inv
            partB = 0.5 * np.log((alpha_l[:,np.newaxis] * tr_tr_sums * alpha_l).sum())
            return partA + partB + partC

        # the initial point is set as the uniform distribution
        uniform_distribution = np.full(fill_value=1 / n, shape=(n,))

        # solutions are bounded to those contained in the unit-simplex
        bounds = tuple((0, 1) for _ in range(n))  # values in [0,1]
        constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})  # values summing up to 1
        r = optimize.minimize(match, x0=uniform_distribution, method='SLSQP', bounds=bounds, constraints=constraints)
        return r.x


