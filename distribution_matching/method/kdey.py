from typing import Union
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity

import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import AggregativeProbabilisticQuantifier, cross_generate_predictions
import quapy.functional as F

from sklearn.metrics.pairwise import rbf_kernel


class KDEBase:

    BANDWIDTH_METHOD = ['scott', 'silverman']

    @classmethod
    def _check_bandwidth(cls, bandwidth):
        assert bandwidth in KDEBase.BANDWIDTH_METHOD or isinstance(bandwidth, float), \
            f'invalid bandwidth, valid ones are {KDEBase.BANDWIDTH_METHOD} or float values'
        if isinstance(bandwidth, float):
            assert 0 < bandwidth < 1,  "the bandwith for KDEy should be in (0,1), since this method models the unit simplex"

    def get_kde_function(self, X, bandwidth):
        return KernelDensity(bandwidth=bandwidth).fit(X)

    def pdf(self, kde, X):
        return np.exp(kde.score_samples(X))

    def get_mixture_components(self, X, y, n_classes, bandwidth):
        return [self.get_kde_function(X[y == cat], bandwidth) for cat in range(n_classes)]



class KDEyML(AggregativeProbabilisticQuantifier, KDEBase):

    def __init__(self, classifier: BaseEstimator, val_split=10, bandwidth=0.1, n_jobs=None, random_state=0):
        self._check_bandwidth(bandwidth)
        self.classifier = classifier
        self.val_split = val_split
        self.bandwidth = bandwidth
        self.n_jobs = n_jobs
        self.random_state=random_state

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split: Union[float, LabelledCollection] = None):
        if val_split is None:
            val_split = self.val_split

        self.classifier, y, posteriors, _, _ = cross_generate_predictions(
            data, self.classifier, val_split, probabilistic=True, fit_classifier=fit_classifier, n_jobs=self.n_jobs
        )

        self.mix_densities = self.get_mixture_components(posteriors, y, data.n_classes, self.bandwidth)

        return self

    def aggregate(self, posteriors: np.ndarray):
        """
        Searches for the mixture model parameter (the sought prevalence values) that maximizes the likelihood
        of the data (i.e., that minimizes the negative log-likelihood)

        :param posteriors: instances in the sample converted into posterior probabilities
        :return: a vector of class prevalence estimates
        """
        np.random.RandomState(self.random_state)
        epsilon = 1e-10
        n_classes = len(self.mix_densities)
        test_densities = [self.pdf(kde_i, posteriors) for kde_i in self.mix_densities]

        def neg_loglikelihood(prev):
            test_mixture_likelihood = sum(prev_i * dens_i for prev_i, dens_i in zip (prev, test_densities))
            test_loglikelihood = np.log(test_mixture_likelihood + epsilon)
            return  -np.sum(test_loglikelihood)

        return F.optim_minimize(neg_loglikelihood, n_classes)


class KDEyHD(AggregativeProbabilisticQuantifier, KDEBase):

    def __init__(self, classifier: BaseEstimator, val_split=10, divergence: str='HD',
                 bandwidth=0.1, n_jobs=None, random_state=0, montecarlo_trials=10000):
        
        self._check_bandwidth(bandwidth)
        self.classifier = classifier
        self.val_split = val_split
        self.divergence = divergence
        self.bandwidth = bandwidth
        self.n_jobs = n_jobs
        self.random_state=random_state
        self.montecarlo_trials = montecarlo_trials

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split: Union[float, LabelledCollection] = None):
        if val_split is None:
            val_split = self.val_split

        self.classifier, y, posteriors, _, _ = cross_generate_predictions(
            data, self.classifier, val_split, probabilistic=True, fit_classifier=fit_classifier, n_jobs=self.n_jobs
        )

        self.mix_densities = self.get_mixture_components(posteriors, y, data.n_classes, self.bandwidth)

        N = self.montecarlo_trials
        rs = self.random_state
        n = data.n_classes
        self.reference_samples = np.vstack([kde_i.sample(N//n, random_state=rs) for kde_i in self.mix_densities])
        self.reference_classwise_densities = np.asarray([self.pdf(kde_j, self.reference_samples) for kde_j in self.mix_densities])
        self.reference_density = np.mean(self.reference_classwise_densities, axis=0)  # equiv. to (uniform @ self.reference_classwise_densities)

        return self

    def aggregate(self, posteriors: np.ndarray):
        # we retain all n*N examples (sampled from a mixture with uniform parameter), and then
        # apply importance sampling (IS). In this version we compute D(p_alpha||q) with IS
        n_classes = len(self.mix_densities)

        test_kde = self.get_kde_function(posteriors, self.bandwidth)
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


class KDEyCS(AggregativeProbabilisticQuantifier):

    def __init__(self, classifier: BaseEstimator, val_split=10, bandwidth=0.1, n_jobs=None, random_state=0):
        KDEBase._check_bandwidth(bandwidth)
        self.classifier = classifier
        self.val_split = val_split
        self.bandwidth = bandwidth
        self.n_jobs = n_jobs
        self.random_state=random_state

    def gram_matrix_mix_sum(self, X, Y=None):
        # this adapts the output of the rbf_kernel function (pairwise evaluations of Gaussian kernels k(x,y))
        # to contain pairwise evaluations of N(x|mu,Sigma1+Sigma2) with mu=y and Sigma1 and Sigma2 are 
        # two "scalar matrices" (h^2)*I each, so Sigma1+Sigma2 has scalar 2(h^2) (h is the bandwidth)
        h = self.bandwidth
        variance = 2 * (h**2)
        nD = X.shape[1]
        gamma = 1/(2*variance)
        norm_factor = 1/np.sqrt(((2*np.pi)**nD) * (variance**(nD)))
        gram = norm_factor * rbf_kernel(X, Y, gamma=gamma)
        return gram.sum()

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split: Union[float, LabelledCollection] = None):
        if val_split is None:
            val_split = self.val_split

        self.classifier, y, posteriors, _, _ = cross_generate_predictions(
            data, self.classifier, val_split, probabilistic=True, fit_classifier=fit_classifier, n_jobs=self.n_jobs
        )

        assert all(sorted(np.unique(y)) == np.arange(data.n_classes)), \
            'label name gaps not allowed in current implementation'

        n = data.n_classes
        P = posteriors

        # counts_inv keeps track of the relative weight of each datapoint within its class
        # (i.e., the weight in its KDE model)
        counts_inv = 1 / (data.counts())

        # tr_tr_sums corresponds to symbol \overline{B} in the paper
        tr_tr_sums = np.zeros(shape=(n,n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i > j:
                    tr_tr_sums[i,j] = tr_tr_sums[j,i]
                else:
                    block = self.gram_matrix_mix_sum(P[y == i], P[y == j] if i!=j else None)
                    tr_tr_sums[i, j] = block

        # keep track of these data structures for the test phase
        self.Ptr = P
        self.ytr = y
        self.tr_tr_sums = tr_tr_sums
        self.counts_inv = counts_inv

        return self


    def aggregate(self, posteriors: np.ndarray):
        Ptr = self.Ptr
        Pte = posteriors
        y = self.ytr
        tr_tr_sums = self.tr_tr_sums

        M, nD = Pte.shape
        Minv = (1/M) # t in the paper
        n = Ptr.shape[1]


        # becomes a constant that does not affect the optimization, no need to compute it
        # partC = 0.5*np.log(self.gram_matrix_mix_sum(Pte) * Kinv * Kinv)

        # tr_te_sums corresponds to \overline{a}*(1/Li)*(1/M) in the paper (note the constants
        # are already aggregated to tr_te_sums, so these multiplications are not carried out
        # at each iteration of the optimization phase)
        tr_te_sums = np.zeros(shape=n, dtype=float)
        for i in range(n):
            tr_te_sums[i] = self.gram_matrix_mix_sum(Ptr[y==i], Pte) 

        def divergence(alpha):
            # called \overline{r} in the paper
            alpha_ratio = alpha * self.counts_inv

            # recal that tr_te_sums already accounts for the constant terms (1/Li)*(1/M)
            partA = -np.log((alpha_ratio @ tr_te_sums) * Minv)
            partB = 0.5 * np.log(alpha_ratio @ tr_tr_sums @ alpha_ratio)
            return partA + partB #+ partC

        return F.optim_minimize(divergence, n)

