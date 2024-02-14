from typing import Union
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity

import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import AggregativeSoftQuantifier
import quapy.functional as F

from sklearn.metrics.pairwise import rbf_kernel


class KDEBase:
    """
    Common ancestor for KDE-based methods. Implements some common routines.
    """

    BANDWIDTH_METHOD = ['scott', 'silverman']

    @classmethod
    def _check_bandwidth(cls, bandwidth):
        """
        Checks that the bandwidth parameter is correct

        :param bandwidth: either a string (see BANDWIDTH_METHOD) or a float
        :return: nothing, but raises an exception for invalid values
        """
        assert bandwidth in KDEBase.BANDWIDTH_METHOD or isinstance(bandwidth, float), \
            f'invalid bandwidth, valid ones are {KDEBase.BANDWIDTH_METHOD} or float values'
        if isinstance(bandwidth, float):
            assert 0 < bandwidth < 1,  "the bandwith for KDEy should be in (0,1), since this method models the unit simplex"

    def get_kde_function(self, X, bandwidth):
        """
        Wraps the KDE function from scikit-learn.

        :param X: data for which the density function is to be estimated
        :param bandwidth: the bandwidth of the kernel
        :return: a scikit-learn's KernelDensity object
        """
        return KernelDensity(bandwidth=bandwidth).fit(X)

    def pdf(self, kde, X):
        """
        Wraps the density evalution of scikit-learn's KDE. Scikit-learn returns log-scores (s), so this
        function returns :math:`e^{s}`

        :param kde: a previously fit KDE function
        :param X: the data for which the density is to be estimated
        :return: np.ndarray with the densities
        """
        return np.exp(kde.score_samples(X))

    def get_mixture_components(self, X, y, n_classes, bandwidth):
        """
        Returns an array containing the mixture components, i.e., the KDE functions for each class.

        :param X: the data containing the covariates
        :param y: the class labels
        :param n_classes: integer, the number of classes
        :param bandwidth: float, the bandwidth of the kernel
        :return: a list of KernelDensity objects, each fitted with the corresponding class-specific covariates
        """
        return [self.get_kde_function(X[y == cat], bandwidth) for cat in range(n_classes)]



class KDEyML(AggregativeSoftQuantifier, KDEBase):
    """
    Kernel Density Estimation model for quantification (KDEy) relying on the Kullback-Leibler divergence (KLD) as
    the divergence measure to be minimized. This method was first proposed in the paper
    `Kernel Density Estimation for Multiclass Quantification <https://arxiv.org/abs/2401.00490>`_, in which
    the authors show that minimizing the distribution mathing criterion for KLD is akin to performing
    maximum likelihood (ML).

    The distribution matching optimization problem comes down to solving:

    :math:`\\hat{\\alpha} = \\arg\\min_{\\alpha\\in\\Delta^{n-1}} \\mathcal{D}(\\boldsymbol{p}_{\\alpha}||q_{\\widetilde{U}})`

    where :math:`p_{\\alpha}` is the mixture of class-specific KDEs with mixture parameter (hence class prevalence)
    :math:`\\alpha` defined by

    :math:`\\boldsymbol{p}_{\\alpha}(\\widetilde{x}) = \\sum_{i=1}^n \\alpha_i p_{\\widetilde{L}_i}(\\widetilde{x})`

    where :math:`p_X(\\boldsymbol{x}) = \\frac{1}{|X|} \\sum_{x_i\\in X} K\\left(\\frac{x-x_i}{h}\\right)` is the
    KDE function that uses the datapoints in X as the kernel centers.

    In KDEy-ML, the divergence is taken to be the Kullback-Leibler Divergence. This is equivalent to solving:
    :math:`\\hat{\\alpha} = \\arg\\min_{\\alpha\\in\\Delta^{n-1}} -
    \\mathbb{E}_{q_{\\widetilde{U}}} \\left[ \\log \\boldsymbol{p}_{\\alpha}(\\widetilde{x}) \\right]`

    which corresponds to the maximum likelihood estimate.

    :param classifier: a sklearn's Estimator that generates a binary classifier.
    :param val_split: specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer (default 5), indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`); or as a collection defining the specific set of data to use for validation.
        Alternatively, this set can be specified at fit time by indicating the exact set of data
        on which the predictions are to be generated.
    :param bandwidth: float, the bandwidth of the Kernel
    :param n_jobs: number of parallel workers
    :param random_state: a seed to be set before fitting any base quantifier (default None)
    """

    def __init__(self, classifier: BaseEstimator, val_split=10, bandwidth=0.1, n_jobs=None, random_state=None):
        self._check_bandwidth(bandwidth)
        self.classifier = classifier
        self.val_split = val_split
        self.bandwidth = bandwidth
        self.n_jobs = n_jobs
        self.random_state=random_state

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        self.mix_densities = self.get_mixture_components(*classif_predictions.Xy, data.n_classes, self.bandwidth)
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


class KDEyHD(AggregativeSoftQuantifier, KDEBase):
    """
    Kernel Density Estimation model for quantification (KDEy) relying on the squared Hellinger Disntace (HD) as
    the divergence measure to be minimized. This method was first proposed in the paper
    `Kernel Density Estimation for Multiclass Quantification <https://arxiv.org/abs/2401.00490>`_, in which
    the authors proposed a Monte Carlo approach for minimizing the divergence.

    The distribution matching optimization problem comes down to solving:

    :math:`\\hat{\\alpha} = \\arg\\min_{\\alpha\\in\\Delta^{n-1}} \\mathcal{D}(\\boldsymbol{p}_{\\alpha}||q_{\\widetilde{U}})`

    where :math:`p_{\\alpha}` is the mixture of class-specific KDEs with mixture parameter (hence class prevalence)
    :math:`\\alpha` defined by

    :math:`\\boldsymbol{p}_{\\alpha}(\\widetilde{x}) = \\sum_{i=1}^n \\alpha_i p_{\\widetilde{L}_i}(\\widetilde{x})`

    where :math:`p_X(\\boldsymbol{x}) = \\frac{1}{|X|} \\sum_{x_i\\in X} K\\left(\\frac{x-x_i}{h}\\right)` is the
    KDE function that uses the datapoints in X as the kernel centers.

    In KDEy-HD, the divergence is taken to be the squared Hellinger Distance, an f-divergence with corresponding
    f-generator function given by:

    :math:`f(u)=(\\sqrt{u}-1)^2`

    The authors proposed a Monte Carlo solution that relies on importance sampling:

    :math:`\\hat{D}_f(p||q)= \\frac{1}{t} \\sum_{i=1}^t f\\left(\\frac{p(x_i)}{q(x_i)}\\right) \\frac{q(x_i)}{r(x_i)}`

    where the datapoints (trials) :math:`x_1,\\ldots,x_t\\sim_{\\mathrm{iid}} r` with :math:`r`  the
    uniform distribution.

    :param classifier: a sklearn's Estimator that generates a binary classifier.
    :param val_split: specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer (default 5), indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`); or as a collection defining the specific set of data to use for validation.
        Alternatively, this set can be specified at fit time by indicating the exact set of data
        on which the predictions are to be generated.
    :param bandwidth: float, the bandwidth of the Kernel
    :param n_jobs: number of parallel workers
    :param random_state: a seed to be set before fitting any base quantifier (default None)
    :param montecarlo_trials: number of Monte Carlo trials (default 10000)
    """

    def __init__(self, classifier: BaseEstimator, val_split=10, divergence: str='HD',
                 bandwidth=0.1, n_jobs=None, random_state=None, montecarlo_trials=10000):
        
        self._check_bandwidth(bandwidth)
        self.classifier = classifier
        self.val_split = val_split
        self.divergence = divergence
        self.bandwidth = bandwidth
        self.n_jobs = n_jobs
        self.random_state=random_state
        self.montecarlo_trials = montecarlo_trials

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        self.mix_densities = self.get_mixture_components(*classif_predictions.Xy, data.n_classes, self.bandwidth)

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


class KDEyCS(AggregativeSoftQuantifier):
    """
    Kernel Density Estimation model for quantification (KDEy) relying on the Cauchy-Schwarz divergence (CS) as
    the divergence measure to be minimized. This method was first proposed in the paper
    `Kernel Density Estimation for Multiclass Quantification <https://arxiv.org/abs/2401.00490>`_, in which
    the authors proposed a Monte Carlo approach for minimizing the divergence.

    The distribution matching optimization problem comes down to solving:

    :math:`\\hat{\\alpha} = \\arg\\min_{\\alpha\\in\\Delta^{n-1}} \\mathcal{D}(\\boldsymbol{p}_{\\alpha}||q_{\\widetilde{U}})`

    where :math:`p_{\\alpha}` is the mixture of class-specific KDEs with mixture parameter (hence class prevalence)
    :math:`\\alpha` defined by

    :math:`\\boldsymbol{p}_{\\alpha}(\\widetilde{x}) = \\sum_{i=1}^n \\alpha_i p_{\\widetilde{L}_i}(\\widetilde{x})`

    where :math:`p_X(\\boldsymbol{x}) = \\frac{1}{|X|} \\sum_{x_i\\in X} K\\left(\\frac{x-x_i}{h}\\right)` is the
    KDE function that uses the datapoints in X as the kernel centers.

    In KDEy-CS, the divergence is taken to be the Cauchy-Schwarz divergence given by:

    :math:`\\mathcal{D}_{\\mathrm{CS}}(p||q)=-\\log\\left(\\frac{\\int p(x)q(x)dx}{\\sqrt{\\int p(x)^2dx \\int q(x)^2dx}}\\right)`

    The authors showed that this distribution matching admits a closed-form solution

    :param classifier: a sklearn's Estimator that generates a binary classifier.
    :param val_split: specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer (default 5), indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`); or as a collection defining the specific set of data to use for validation.
        Alternatively, this set can be specified at fit time by indicating the exact set of data
        on which the predictions are to be generated.
    :param bandwidth: float, the bandwidth of the Kernel
    :param n_jobs: number of parallel workers
    """

    def __init__(self, classifier: BaseEstimator, val_split=10, bandwidth=0.1, n_jobs=None):
        KDEBase._check_bandwidth(bandwidth)
        self.classifier = classifier
        self.val_split = val_split
        self.bandwidth = bandwidth
        self.n_jobs = n_jobs

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

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):

        P, y = classif_predictions.Xy
        n = data.n_classes

        assert all(sorted(np.unique(y)) == np.arange(n)), \
            'label name gaps not allowed in current implementation'

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

