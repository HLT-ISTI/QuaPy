import os
import sys
from typing import Union, Callable
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

import quapy as qp
from quapy.data import LabelledCollection
from quapy.protocol import APP, UPP
from quapy.method.aggregative import AggregativeProbabilisticQuantifier, _training_helper, cross_generate_predictions, \
    DistributionMatching, _get_divergence
import scipy
from scipy import optimize
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional


# TODO: optimize the bandwidth automatically
# TODO: think of a MMD-y variant, i.e., a MMD variant that uses the points in the simplex and possibly any non-linear kernel


class KDEy(AggregativeProbabilisticQuantifier):

    BANDWIDTH_METHOD = ['auto', 'scott', 'silverman']
    ENGINE = ['scipy', 'sklearn', 'statsmodels']
    TARGET = ['min_divergence', 'max_likelihood']

    def __init__(self, classifier: BaseEstimator, val_split=0.4, divergence: Union[str, Callable]='L2',
                 bandwidth='scott', engine='sklearn', target='min_divergence', n_jobs=None, random_state=0, montecarlo_trials=1000):
        assert bandwidth in KDEy.BANDWIDTH_METHOD or isinstance(bandwidth, float), \
            f'unknown bandwidth_method, valid ones are {KDEy.BANDWIDTH_METHOD}'
        assert engine in KDEy.ENGINE, f'unknown engine, valid ones are {KDEy.ENGINE}'
        assert target in KDEy.TARGET, f'unknown target, valid ones are {KDEy.TARGET}'
        self.classifier = classifier
        self.val_split = val_split
        self.divergence = divergence
        self.bandwidth = bandwidth
        self.engine = engine
        self.target = target
        self.n_jobs = n_jobs
        self.random_state=random_state
        self.montecarlo_trials = montecarlo_trials

    def search_bandwidth_maxlikelihood(self, posteriors, labels):
        grid = {'bandwidth': np.linspace(0.001, 0.2, 100)}
        search = GridSearchCV(
            KernelDensity(), param_grid=grid, n_jobs=-1, cv=50, verbose=1, refit=True
        )
        search.fit(posteriors, labels)
        bandwidth = search.best_params_['bandwidth']
        print(f'auto: bandwidth={bandwidth:.5f}')
        return bandwidth

    def get_kde(self, posteriors):
        # if self.bandwidth == 'auto':
        #     print('adjusting bandwidth')
        #
        #     if self.engine == 'sklearn':
        #         grid = {'bandwidth': np.linspace(0.001,0.2,41)}
        #         search = GridSearchCV(
        #             KernelDensity(), param_grid=grid, n_jobs=-1, cv=10, verbose=1, refit=True
        #         )
        #         search.fit(posteriors)
        #         print(search.best_score_)
        #         print(search.best_params_)
        #
        #         import pandas as pd
        #         df = pd.DataFrame(search.cv_results_)
        #         pd.set_option('display.max_columns', None)
        #         pd.set_option('display.max_rows', None)
        #         pd.set_option('expand_frame_repr', False)
        #         print(df)
        #         sys.exit(0)
        #
        #         kde = search

        #else:
        if self.engine == 'scipy':
            # scipy treats columns as datapoints, and need the datapoints not to lie in a lower-dimensional subspace, which
            # requires removing the last dimension which is constrained
            posteriors = posteriors[:,:-1].T
            kde = scipy.stats.gaussian_kde(posteriors)
            kde.set_bandwidth(self.bandwidth)
        elif self.engine == 'sklearn':
            #print('fitting kde')
            kde = KernelDensity(bandwidth=self.bandwidth).fit(posteriors)
            #print('[fitting done]')
        return kde

    def pdf(self, kde, posteriors):
        if self.engine == 'scipy':
            return kde(posteriors[:, :-1].T)
        elif self.engine == 'sklearn':
            #print('pdf...')
            densities = np.exp(kde.score_samples(posteriors))
            #print('[pdf done]')
            return densities
            #return np.exp(kde.score_samples(posteriors))

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split: Union[float, LabelledCollection] = None):
        """

        :param data: the training set
        :param fit_classifier: set to False to bypass the training (the learner is assumed to be already fit)
        :param val_split: either a float in (0,1) indicating the proportion of training instances to use for
         validation (e.g., 0.3 for using 30% of the training set as validation data), or a LabelledCollection
         indicating the validation set itself, or an int indicating the number k of folds to be used in kFCV
         to estimate the parameters
        """
        if val_split is None:
            val_split = self.val_split

        self.classifier, y, posteriors, classes, class_count = cross_generate_predictions(
            data, self.classifier, val_split, probabilistic=True, fit_classifier=fit_classifier, n_jobs=self.n_jobs
        )

        if self.bandwidth == 'auto':
            self.bandwidth = self.search_bandwidth_maxlikelihood(posteriors, y)

        self.val_densities = [self.get_kde(posteriors[y == cat]) for cat in range(data.n_classes)]
        self.val_posteriors = posteriors

        if self.target == 'min_divergence':
            self.samples = qp.functional.uniform_prevalence_sampling(n_classes=data.n_classes, size=self.montecarlo_trials)
            self.sample_densities = [self.pdf(kde_i, self.samples) for kde_i in self.val_densities]

        return self

    #def val_pdf(self, prev):
        """
        Returns a function that computes the mixture model with the given prev as mixture factor
        :param prev: a prevalence vector, ndarray
        :return: a function implementing the validation distribution with fixed mixture factor
        """
    #    return lambda posteriors: sum(prev_i * self.pdf(kde_i, posteriors) for kde_i, prev_i in zip(self.val_densities, prev))

    def aggregate(self, posteriors: np.ndarray):
        if self.target == 'min_divergence':
            return self._target_divergence(posteriors)
        elif self.target == 'max_likelihood':
            return self._target_likelihood(posteriors)
        else:
            raise ValueError('unknown target')

    def _target_divergence_depr(self, posteriors):
        # this variant is, I think, ill-formed, since it evaluates the likelihood on the test points, which are
        # overconfident in the KDE-test.        
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

    def _target_divergence(self, posteriors):
        # in this variant we evaluate the divergence using a Montecarlo approach
        n_classes = len(self.val_densities)

        test_kde = self.get_kde(posteriors)
        test_likelihood = self.pdf(test_kde, self.samples)
        
        divergence = _get_divergence(self.divergence)
  
        def match(prev):
            val_likelihood = sum(prev_i * dens_i for prev_i, dens_i in zip (prev, self.sample_densities))
            return divergence(val_likelihood, test_likelihood)
            
        # the initial point is set as the uniform distribution
        uniform_distribution = np.full(fill_value=1 / n_classes, shape=(n_classes,))

        # solutions are bounded to those contained in the unit-simplex
        bounds = tuple((0, 1) for _ in range(n_classes))  # values in [0,1]
        constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})  # values summing up to 1
        r = optimize.minimize(match, x0=uniform_distribution, method='SLSQP', bounds=bounds, constraints=constraints)
        return r.x

    def _target_likelihood(self, posteriors, eps=0.000001):
        """
        Searches for the mixture model parameter (the sought prevalence values) that yields a validation distribution
        (the mixture) that best matches the test distribution, in terms of the divergence measure of choice.

        :param instances: instances in the sample
        :return: a vector of class prevalence estimates
        """
        np.random.RandomState(self.random_state)
        n_classes = len(self.val_densities)
        test_densities = [self.pdf(kde_i, posteriors) for kde_i in self.val_densities]

        #return lambda posteriors: sum(prev_i * self.pdf(kde_i, posteriors) for kde_i, prev_i in zip(self.val_densities, prev))
        def neg_loglikelihood(prev):
            #print('-neg_likelihood')
            #val_pdf = self.val_pdf(prev)
            #test_likelihood = val_pdf(posteriors)
            test_mixture_likelihood = sum(prev_i * dens_i for prev_i, dens_i in zip (prev, test_densities))
            test_loglikelihood = np.log(test_mixture_likelihood + eps)
            neg_log_likelihood = -np.sum(test_loglikelihood)
            #print('-neg_likelihood [done!]')
            return neg_log_likelihood
            #return -np.prod(test_likelihood)

        # the initial point is set as the uniform distribution
        uniform_distribution = np.full(fill_value=1 / n_classes, shape=(n_classes,))

        # solutions are bounded to those contained in the unit-simplex
        bounds = tuple((0, 1) for _ in range(n_classes))  # values in [0,1]
        constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})  # values summing up to 1
        #print('searching for alpha')
        r = optimize.minimize(neg_loglikelihood, x0=uniform_distribution, method='SLSQP', bounds=bounds, constraints=constraints)
        #print('[optimization ended]')
        return r.x