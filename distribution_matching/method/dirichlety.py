import os
import sys
from typing import Union, Callable
import numpy as np
from dirichlet.dirichlet import NotConvergingError
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

import quapy as qp
from quapy.data import LabelledCollection
from quapy.protocol import APP, UPP
from quapy.method.aggregative import AggregativeProbabilisticQuantifier, _training_helper, cross_generate_predictions, \
    DistributionMatching, _get_divergence, CC, PCC
import scipy
from scipy import optimize
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional
import dirichlet


class DIRy(AggregativeProbabilisticQuantifier):

    MAXITER = 100000

    def __init__(self, classifier: BaseEstimator, val_split=0.4, n_jobs=None, target='max_likelihood'):
        self.classifier = classifier
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.target = target
        self.dummy = None

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split: Union[float, LabelledCollection] = None):

        if val_split is None:
            val_split = self.val_split

        self.classifier, y, posteriors, _, _ = cross_generate_predictions(
            data, self.classifier, val_split, probabilistic=True, fit_classifier=fit_classifier, n_jobs=self.n_jobs
        )

        self.val_parameters = []
        try:
            for cat in range(data.n_classes):
                dir_i = dirichlet.mle(posteriors[y == cat], maxiter=DIRy.MAXITER)
                self.val_parameters.append(dir_i)
        except NotConvergingError as e:
            print(e)
            print(f'{self.__class__} failed to converge; resorting to PCC')
            self.dummy = PCC(self.classifier).fit(data, fit_classifier=fit_classifier)

        return self

    def val_pdf(self, prev):
        """
        Returns a function that computes the mixture model with the given prev as mixture factor
        :param prev: a prevalence vector, ndarray
        :return: a function implementing the validation distribution with fixed mixture factor
        """
        return lambda posteriors: sum(prev_i * dirichlet.pdf(parameters_i)(posteriors) for parameters_i, prev_i in zip(self.val_parameters, prev))

    def aggregate(self, posteriors: np.ndarray):
        if self.dummy is not None:
            return self.dummy.aggregate(posteriors)

        if self.target == 'min_divergence':
            raise NotImplementedError('not yet')
            return self._target_divergence(posteriors)
        elif self.target == 'max_likelihood':
            return self._target_likelihood(posteriors)
        else:
            raise ValueError('unknown target')

    def _target_divergence(self, posteriors):
        test_density = self.get_kde(posteriors)
        # val_test_posteriors = np.concatenate([self.val_posteriors, posteriors])
        test_likelihood = self.pdf(test_density, posteriors)
        divergence = _get_divergence(self.divergence)

        n_classes = len(self.val_densities)

        def match(prev):
            val_pdf = self.val_pdf(prev)
            val_likelihood = val_pdf(posteriors)
            return divergence(val_likelihood, test_likelihood)

        # the initial point is set as the uniform distribution
        uniform_distribution = np.full(fill_value=1 / n_classes, shape=(n_classes,))

        # solutions are bounded to those contained in the unit-simplex
        bounds = tuple((0, 1) for _ in range(n_classes))  # values in [0,1]
        constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})  # values summing up to 1
        r = optimize.minimize(match, x0=uniform_distribution, method='SLSQP', bounds=bounds, constraints=constraints)
        return r.x

    def _target_likelihood(self, posteriors, eps=0.000001):
        n_classes = len(self.val_parameters)

        def neg_loglikelihood(prev):
            val_pdf = self.val_pdf(prev)
            test_likelihood = val_pdf(posteriors)
            test_loglikelihood = np.log(test_likelihood + eps)
            return -np.sum(test_loglikelihood)

        # the initial point is set as the uniform distribution
        uniform_distribution = np.full(fill_value=1 / n_classes, shape=(n_classes,))

        # solutions are bounded to those contained in the unit-simplex
        bounds = tuple((0, 1) for _ in range(n_classes))  # values in [0,1]
        constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})  # values summing up to 1
        r = optimize.minimize(neg_loglikelihood, x0=uniform_distribution, method='SLSQP', bounds=bounds, constraints=constraints)
        return r.x