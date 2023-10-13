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


def gram_matrix_mix(bandwidth, X, Y=None):
    # this adapts the output of the rbf_kernel function (pairwise evaluations of Gaussian kernels k(x,y))
    # to contain pairwise evaluations of N(x|mu,Sigma1+Sigma2) with mu=y and Sigma1 and Sigma2 are 
    # two "scalar matrices" (h^2) I each, so Sigma1+Sigma2 has scalar 2(h^2) (h is the bandwidth)
    variance = 2 * (bandwidth**2)
    nD = X.shape[1]
    gamma = 1/(2*variance)
    gram = rbf_kernel(X, Y, gamma=gamma)
    norm_factor = 1/np.sqrt(((2*np.pi)**nD) * (variance**(nD)))
    gram *= norm_factor
    print('GRAM SUM:', gram.sum())
    return gram

def weighted_prod(pi, tau, G):
    return pi[:,np.newaxis] * G * tau

def tril_weighted_prod(pi, G):
    M = weighted_prod(pi, pi, G)
    return np.triu(M,1)


class KDEyclosed(AggregativeProbabilisticQuantifier):

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

        # from distribution_matching.binary_debug import HACK
        # posteriors, y = HACK(posteriors, y)

        # print('[fit] precomputing stuff')

        n = data.n_classes
        #L = [posteriors[y==i] for i in range(n)]
        #l = [len(Li) for Li in L]

        D = n
        h = self.bandwidth
        #cov_mix_scalar = 2 * h * h  # corresponds to a bandwidth of sqrt(2)*h
        #Kernel = multivariate_normal(mean=np.zeros(D), cov=cov_mix_scalar)

        # print('[fit] classifier ready; precomputing gram')
        self.gram_tr_tr = gram_matrix_mix(h, posteriors)

        # li_inv keeps track of the relative weight of each datapoint within its class 
        # (i.e., the weight in its KDE model)
        counts_inv = 1/(data.counts())
        self.li_inv = counts_inv[y]

#        Khash = {}
#        for a in range(n):
#            for b in range(l[a]):
#                for i in range(n):
#                    Khash[(a,b,i)] = sum(Kernel.pdf(L[i][j] - L[a][b]) for j in range(l[i]))
                    # for j in range(l[i]):  # this for, and index j, can be supressed and store the sum across j
                    #     Khash[(a, b, i, j)] = Kernel.pdf(L[i][j] - L[a][b])

        self.n = n
        #self.L = L
        #self.l = l
        #self.Kernel = Kernel
        #self.Khash = Khash
        self.C = ((2 * np.pi) ** (-D / 2)) * h ** (-D)
        print('C:', self.C)
        self.Ptr = posteriors
        self.ytr = y

        assert all(sorted(np.unique(y)) == np.arange(data.n_classes)), 'label name gaps not allowed in current implementation'

        # print('[fit] exit')

        return self


    def aggregate(self, posteriors: np.ndarray):

        # print('[aggregate] enter')

        Ptr = self.Ptr
        Pte = posteriors

        gram_te_te = gram_matrix_mix(self.bandwidth, Pte, Pte)
        gram_tr_te = gram_matrix_mix(self.bandwidth, Ptr, Pte)

        K = Pte.shape[0]
        tau = np.full(shape=K, fill_value=1/K, dtype=float)

        h = self.bandwidth
        D = Ptr.shape[1]
        C = self.C

        partC = 0.5 * np.log( C/K + 2 * tril_weighted_prod(tau, gram_te_te).sum())
        
        def match(alpha):

            pi = alpha[self.ytr] * self.li_inv

            partA = -np.log(weighted_prod(pi, tau, gram_tr_te).sum())
            # print('gram_Tr_Tr sum', self.gram_tr_tr.sum())
            # print('pretril', (np.triu(self.gram_tr_tr,1).sum()))
            # print('tril', (2 * tril_weighted_prod(pi, self.gram_tr_tr).sum()))
            # print('pi', pi.sum(), pi[:10])
            # print('Cs', C*(pi**2).sum())
            partB = 0.5 * np.log(C*(pi**2).sum() + 2*tril_weighted_prod(pi, self.gram_tr_tr).sum())

            Dcs = partA + partB + partC

            # print(f'{alpha=}\t{partA=}\t{partB=}\t{partC}')
            # print()

            return Dcs





        # print('[aggregate] starts search')

        # the initial point is set as the uniform distribution
        uniform_distribution = np.full(fill_value=1 / self.n, shape=(self.n,))
        # uniform_distribution = [0.2, 0.8]

        # solutions are bounded to those contained in the unit-simplex
        bounds = tuple((0, 1) for _ in range(self.n))  # values in [0,1]
        constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})  # values summing up to 1
        r = optimize.minimize(match, x0=uniform_distribution, method='SLSQP', bounds=bounds, constraints=constraints)
        # print('[aggregate] end')
        return r.x


