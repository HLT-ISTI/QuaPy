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

import quapy as qp
from quapy.data import LabelledCollection
from quapy.protocol import APP, UPP
from quapy.method.aggregative import AggregativeProbabilisticQuantifier, _training_helper, cross_generate_predictions, \
    DistributionMatching, _get_divergence
import quapy.functional as F
import scipy
from scipy import optimize
#from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional


# TODO: optimize the bandwidth automatically
# TODO: think of a MMD-y variant, i.e., a MMD variant that uses the points in the simplex and possibly any non-linear kernel


class KDEy(AggregativeProbabilisticQuantifier):

    BANDWIDTH_METHOD = ['auto', 'scott', 'silverman']
    ENGINE = ['scipy', 'sklearn', 'statsmodels']
    TARGET = ['min_divergence', 'min_divergence_uniform', 'max_likelihood']

    def __init__(self, classifier: BaseEstimator, val_split=0.4, divergence: Union[str, Callable]='L2',
                 bandwidth='scott', engine='sklearn', target='min_divergence', n_jobs=None, random_state=0, montecarlo_trials=1000):
        assert bandwidth in KDEy.BANDWIDTH_METHOD or isinstance(bandwidth, float), \
            f'unknown bandwidth_method, valid ones are {KDEy.BANDWIDTH_METHOD}'
        assert engine in KDEy.ENGINE, f'unknown engine, valid ones are {KDEy.ENGINE}'
        assert target in KDEy.TARGET, f'unknown target, valid ones are {KDEy.TARGET}'
        assert target=='max_likelihood' or divergence in ['KLD', 'HD', 'JS'], \
            'in this version I will only allow KLD or squared HD as a divergence'
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

    def get_kde_function(self, posteriors):
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
            return np.exp(kde.score_samples(posteriors))

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

        self.val_densities = [self.get_kde_function(posteriors[y == cat]) for cat in range(data.n_classes)]
        self.val_posteriors = posteriors

        if self.target == 'min_divergence_uniform':
            self.samples = qp.functional.uniform_prevalence_sampling(n_classes=data.n_classes, size=self.montecarlo_trials)
            self.sample_densities = [self.pdf(kde_i, self.samples) for kde_i in self.val_densities]
        elif self.target == 'min_divergence':
            N = self.montecarlo_trials
            rs = self.random_state
            n = data.n_classes
            self.reference_samples = np.vstack([kde_i.sample(N//n, random_state=rs) for kde_i in self.val_densities])
            self.reference_classwise_densities = np.asarray([self.pdf(kde_j, self.reference_samples) for kde_j in self.val_densities])
            self.reference_density = np.mean(self.reference_classwise_densities, axis=0)  # equiv. to (uniform @ self.reference_classwise_densities)
        elif self.target == 'min_divergence_deprecated':  # the version of the first draft, with n*N presampled, then alpha*N chosen for class
            self.class_samples = [kde_i.sample(self.montecarlo_trials, random_state=self.random_state) for kde_i in self.val_densities]
            self.class_sample_densities = {}
            for ci, samples_i in enumerate(self.class_samples):
                self.class_sample_densities[ci] = np.asarray([self.pdf(kde_j, samples_i) for kde_j in self.val_densities]).T

        return self

    def aggregate(self, posteriors: np.ndarray):
        if self.target == 'min_divergence':
            return self._target_divergence(posteriors)
        elif self.target == 'min_divergence_uniform':
            return self._target_divergence_uniform(posteriors)
        elif self.target == 'max_likelihood':
            return self._target_likelihood(posteriors)
        else:
            raise ValueError('unknown target')


    # new version in which we retain all n*N examples (sampled from a mixture with uniform parameter), and then
    # apply importance sampling (IS). In this version we compute D(p_alpha||q) with IS, and not D(q||p_alpha) as
    # in the first draft
    def _target_divergence(self, posteriors):
        # in this variant we evaluate the divergence using a Montecarlo approach
        n_classes = len(self.val_densities)

        test_kde = self.get_kde_function(posteriors)
        test_densities = self.pdf(test_kde, self.reference_samples)

        def f_squared_hellinger(u):
            return (np.sqrt(u)-1)**2

        # todo: this will fail when self.divergence is a callable, and is not the right place to do it anyway
        if self.divergence.lower() == 'hd':
            f = f_squared_hellinger
        else:
            raise ValueError('only squared HD is currently implemented')

        epsilon = 1e-10
        qs = test_densities + epsilon
        rs = self.reference_density + epsilon
        iw = qs/rs  #importance weights
        p_class = self.reference_classwise_densities + epsilon
        fracs = p_class/qs

        def divergence(prev):
            # ps / qs = (prev @ p_class) / qs = prev @ (p_class / qs) = prev @ fracs
            ps_div_qs = prev @ fracs
            return np.mean( f(ps_div_qs) * iw )

        return F.optim_minimize(divergence, n_classes)

    # new version in which we retain all n*N examples (sampled from a mixture with uniform parameter), and then
    # apply importance sampling (IS). In this version we compute D(q||p_alpha) with IS, and not D(p_alpha||q) as
    # in the reformulation proposed above
    def _target_divergence_q__p(self, posteriors):
        # in this variant we evaluate the divergence using a Montecarlo approach
        n_classes = len(self.val_densities)

        test_kde = self.get_kde_function(posteriors)
        test_densities = self.pdf(test_kde, self.reference_samples)

        def f_squared_hellinger(u):
            return (np.sqrt(u)-1)**2

        # todo: this will fail when self.divergence is a callable, and is not the right place to do it anyway
        if self.divergence.lower() == 'hd':
            f = f_squared_hellinger
        else:
            raise ValueError('only squared HD is currently implemented')

        epsilon = 1e-10
        qs = test_densities + epsilon
        rs = self.reference_density + epsilon
        p_class = self.reference_classwise_densities + epsilon

        # D(q||p_a) = 1/N sum f(q/p_a) * (p_a / p_u)
        def divergence(prev):
            p_a = prev @ p_class
            return np.mean( f(qs/p_a) * (p_a/rs) )

        return F.optim_minimize(divergence, n_classes)


    # the first version we explain in the draft, choosing alpha*N from a pool of N for each class and w/o importance sampling
    def _target_divergence_deprecated(self, posteriors):
        # in this variant we evaluate the divergence using a Montecarlo approach
        n_classes = len(self.val_densities)

        test_kde = self.get_kde_function(posteriors)
        test_densities_per_class = [self.pdf(test_kde, samples_i) for samples_i in self.class_samples]

        # divergence = _get_divergence(self.divergence)
        def kld_monte(pi, qi, eps=1e-8):
            # there is no pi in front of the log because the samples are already drawn according to pi
            smooth_pi = pi+eps
            smooth_qi = qi+eps
            return np.mean(np.log(smooth_pi / smooth_qi))

        def squared_hellinger(pi, qi, eps=1e-8):
            smooth_pi = pi + eps
            smooth_qi = qi + eps
            return np.mean((np.sqrt(smooth_pi/smooth_qi)-1)**2)

        # todo: this will fail when self.divergence is a callable, and is not the right place to do it anyway
        if self.divergence.lower() == 'kld':
            fdivergence = kld_monte
        elif self.divergence.lower() == 'hd':
            fdivergence = squared_hellinger


        def dissimilarity(prev):
            # choose the samples according to the prevalence vector
            # e.g., prev = [0.5, 0.3, 0.2] will draw 50% from KDE_0, 30% from KDE_1, and 20% from KDE_2
            #       the points are already pre-sampled and de densities are pre-computed, so that now all that remains
            #       is to pick a proportional number of each from each class (same for test)
            num_variates_per_class = np.round(prev * self.montecarlo_trials).astype(int)
            sample_densities = np.vstack(
                [self.class_sample_densities[ci][:num_i] for ci, num_i in enumerate(num_variates_per_class)]
            )
            #val_likelihood = sum(prev_i * dens_i for prev_i, dens_i in zip(prev, sample_densities.T))
            val_likelihood = prev @ sample_densities.T
            #test_likelihood = []
            #for samples_i, num_i in zip(test_densities_per_class, num_variates_per_class):
            #    test_likelihood.append(samples_i[:num_i])
            #test_likelihood = np.concatenate[test]
            test_likelihood = np.concatenate(
                [samples_i[:num_i] for samples_i, num_i in zip(test_densities_per_class, num_variates_per_class)]
            )
            # return fdivergence(val_likelihood, test_likelihood)  # this is wrong, If I sample from the val distribution
            # then I am computing D(Test||Val), so it should be E_{x ~ Val}[f(Test(x)/Val(x))]
            return fdivergence(test_likelihood, val_likelihood)

        return F.optim_minimize(dissimilarity, n_classes)

    # this version explores the entire simplex, and then applies importance sampling. We have not really tested it in deep but
    # seems not to be promising
    def _target_divergence_uniform(self, posteriors):
        # in this variant we evaluate the divergence using a Montecarlo approach
        n_classes = len(self.val_densities)

        test_kde = self.get_kde_function(posteriors)
        test_likelihood = self.pdf(test_kde, self.samples)

        def f_squared_hellinger(t):
            return (np.sqrt(t) - 1)**2

        def f_jensen_shannon(t):
            return -(t+1)*np.log((t+1)/2) + t*np.log(t)

        def fdivergence(pi, qi, f, eps=1e-10):
            spi = pi+eps
            sqi = qi+eps
            return np.mean(f(spi/sqi)*sqi)

        if self.divergence.lower() == 'hd':
            f = f_squared_hellinger
        elif self.divergence.lower() == 'js':
            f = f_jensen_shannon

        def dissimilarity(prev):
            val_likelihood = sum(prev_i * dens_i for prev_i, dens_i in zip (prev, self.sample_densities))
            return fdivergence(val_likelihood, test_likelihood, f)

        return F.optim_minimize(dissimilarity, n_classes)

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
            test_mixture_likelihood = sum(prev_i * dens_i for prev_i, dens_i in zip (prev, test_densities))
            test_loglikelihood = np.log(test_mixture_likelihood + eps)
            return  -np.sum(test_loglikelihood)

        return F.optim_minimize(neg_loglikelihood, n_classes)

