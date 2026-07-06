"""
Utilities and methods for Bayesian quantification.
"""
import contextlib
import copy
import importlib.resources
import logging
import os
import sys
from collections.abc import Iterable
from numbers import Number, Real

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from tqdm import tqdm

import quapy as qp
import quapy.functional as F
from quapy.data import LabelledCollection
from quapy.method._kdey import KDEBase
from quapy.method.aggregative import AggregativeSoftQuantifier
from quapy.method.confidence import ConfidenceRegionABC, WithConfidenceABC
from quapy.protocol import AbstractProtocol

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    from jax.scipy.special import logsumexp as jax_logsumexp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    import stan
    import stan.common

    DEPENDENCIES_INSTALLED = True
except ImportError:
    jax = None
    jnp = None
    jrandom = None
    jax_logsumexp = None
    numpyro = None
    dist = None
    MCMC = None
    NUTS = None
    stan = None

    DEPENDENCIES_INSTALLED = False


P_TEST_Y: str = "P_test(Y)"
P_TEST_C: str = "P_test(C)"
P_C_COND_Y: str = "P(C|Y)"


def _require_bayesian_dependencies():
    if not DEPENDENCIES_INSTALLED:
        raise ImportError(
            "Auxiliary dependencies are required. "
            "Run `$ pip install quapy[bayes]` to install them."
        )


def _resolve_dirichlet_prior(prior, n_classes, *, allow_mapls_priors=False, n_test=None, map_prev=None, map_lambda=None):
    if isinstance(prior, str):
        if prior == 'uniform':
            return np.ones(n_classes, dtype=float)
        if allow_mapls_priors and prior in {'map', 'map2'}:
            if n_test is None or map_prev is None:
                raise ValueError('MAPLS priors require n_test and map_prev')
            if prior == 'map':
                lam = map_lambda
            else:
                lam = get_lambda(
                    test_probs=map_prev["test_probs"],
                    pz=map_prev["train_prev"],
                    q_prior=map_prev["map_estimate"],
                    dvg=kl_div,
                )
            alpha_0 = alpha0_from_lambda(lam, n_test=n_test, n_classes=n_classes)
            return np.full(n_classes, alpha_0, dtype=float)
        raise ValueError(f"unknown prior specification {prior!r}")
    if isinstance(prior, Number):
        return np.full(n_classes, float(prior), dtype=float)

    alpha = np.asarray(prior, dtype=float)
    if alpha.ndim != 1 or len(alpha) != n_classes:
        raise ValueError(f'wrong shape for prior; expected {n_classes} values, found shape {alpha.shape}')
    return alpha


def _validate_temperature(temperature):
    if not isinstance(temperature, Real) or temperature <= 0:
        raise ValueError(f'expected a positive real value for temperature; found {temperature!r}')
    return float(temperature)


def model_bayesianCC(
    n_c_unlabeled: np.ndarray,
    n_y_and_c_labeled: np.ndarray,
    temperature: float,
    alpha: np.ndarray,
) -> None:
    """
    NumPyro model for BayesianCC.
    """
    n_y_labeled = n_y_and_c_labeled.sum(axis=1)

    n_pred_classes = len(n_c_unlabeled)
    n_classes = len(n_y_labeled)

    pi_ = numpyro.sample(P_TEST_Y, dist.Dirichlet(jnp.asarray(alpha, dtype=jnp.float32)))
    p_c_cond_y = numpyro.sample(
        P_C_COND_Y,
        dist.Dirichlet(jnp.ones(n_pred_classes).repeat(n_classes).reshape(n_classes, n_pred_classes)),
    )

    if temperature == 1.0:
        with numpyro.plate('plate', n_classes):
            numpyro.sample('F_yc', dist.Multinomial(n_y_labeled, p_c_cond_y), obs=n_y_and_c_labeled)

        p_c = numpyro.deterministic(P_TEST_C, jnp.einsum("yc,y->c", p_c_cond_y, pi_))
        numpyro.sample('N_c', dist.Multinomial(jnp.sum(n_c_unlabeled), p_c), obs=n_c_unlabeled)
        return

    with numpyro.plate('plate_y', n_classes):
        logp_f = dist.Multinomial(n_y_labeled, p_c_cond_y).log_prob(n_y_and_c_labeled)

    numpyro.factor('F_yc_loglik', jnp.sum(logp_f) / temperature)

    p_c = numpyro.deterministic(P_TEST_C, jnp.einsum("yc,y->c", p_c_cond_y, pi_))
    logp_n = dist.Multinomial(jnp.sum(n_c_unlabeled), p_c).log_prob(n_c_unlabeled)
    numpyro.factor('N_c_loglik', logp_n / temperature)


def model(n_c_unlabeled: np.ndarray, n_y_and_c_labeled: np.ndarray) -> None:
    """
    Backward-compatible BayesianCC model with a uniform prior.
    """
    alpha = np.ones(n_y_and_c_labeled.shape[0], dtype=float)
    return model_bayesianCC(n_c_unlabeled, n_y_and_c_labeled, temperature=1.0, alpha=alpha)


def sample_posterior_bayesianCC(
    n_c_unlabeled: np.ndarray,
    n_y_and_c_labeled: np.ndarray,
    num_warmup: int,
    num_samples: int,
    alpha: np.ndarray,
    temperature: float = 1.0,
    seed: int = 0,
) -> dict:
    """
    Samples from the BayesianCC posterior using NumPyro.
    """
    _require_bayesian_dependencies()
    temperature = _validate_temperature(temperature)

    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(model_bayesianCC),
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=False,
    )
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(
        rng_key,
        n_c_unlabeled=n_c_unlabeled,
        n_y_and_c_labeled=n_y_and_c_labeled,
        temperature=temperature,
        alpha=alpha,
    )
    return mcmc.get_samples()


def sample_posterior(
    n_c_unlabeled: np.ndarray,
    n_y_and_c_labeled: np.ndarray,
    num_warmup: int,
    num_samples: int,
    seed: int = 0,
) -> dict:
    """
    Backward-compatible wrapper around BayesianCC sampling.
    """
    alpha = np.ones(n_y_and_c_labeled.shape[0], dtype=float)
    return sample_posterior_bayesianCC(
        n_c_unlabeled=n_c_unlabeled,
        n_y_and_c_labeled=n_y_and_c_labeled,
        num_warmup=num_warmup,
        num_samples=num_samples,
        alpha=alpha,
        temperature=1.0,
        seed=seed,
    )


def load_stan_file():
    return importlib.resources.files('quapy.method').joinpath('stan/pq.stan').read_text(encoding='utf-8')


@contextlib.contextmanager
def _suppress_stan_logging():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


def pq_stan(stan_code, n_bins, pos_hist, neg_hist, test_hist, number_of_samples, num_warmup, stan_seed):
    """
    Samples posterior prevalences for PQ from a Stan model.
    """
    _require_bayesian_dependencies()
    logging.getLogger("stan.common").setLevel(logging.ERROR)

    stan_data = {
        'n_bucket': n_bins,
        'train_neg': neg_hist.tolist(),
        'train_pos': pos_hist.tolist(),
        'test': test_hist.tolist(),
        'posterior': 1,
    }

    with _suppress_stan_logging():
        stan_model = stan.build(stan_code, data=stan_data, random_seed=stan_seed)
        fit = stan_model.sample(num_chains=1, num_samples=number_of_samples, num_warmup=num_warmup)

    return fit['prev']


class BayesianKDEy(AggregativeSoftQuantifier, KDEBase, WithConfidenceABC):
    """
    Bayesian version of KDEy.

    This method relies on extra dependencies, which have to be installed via:
    `$ pip install quapy[bayes]`

    :param classifier: a scikit-learn's BaseEstimator, or None, in which case
        the classifier is taken to be the one indicated in
        `qp.environ['DEFAULT_CLS']`
    :param fit_classifier: whether to train the classifier, or consider it
        already fit
    :param val_split: specifies the data used for generating classifier
        predictions. This specification can be made as float in (0, 1)
        indicating the proportion of stratified held-out validation set to be
        extracted from the training set; or as an integer (default 5),
        indicating that the predictions are to be generated in a `k`-fold
        cross-validation manner (with this integer indicating the value for
        `k`); or as a tuple `(X,y)` defining the specific set of data to use
        for validation. Set to None when the method does not require any
        validation data, in order to avoid that some portion of the training
        data be wasted.
    :param kernel: kernel function for KDE. Available kernels include
        {'gaussian', 'aitchison', 'ilr'} (default 'gaussian')
    :param bandwidth: bandwidth for the kernel (default 0.1)
    :param shrinkage: regularization strength for Aitchison/ILR kernels
        (default 0.0)
    :param num_warmup: number of warmup iterations for the MCMC sampler
        (default 500)
    :param num_samples: number of samples to draw from the posterior
        (default 1000)
    :param mcmc_seed: random seed for the MCMC sampler (default 0)
    :param confidence_level: float in [0,1] to construct a confidence region
        around the point estimate (default 0.95)
    :param region: string, set to `intervals` for constructing confidence
        intervals (default), or to `ellipse` for constructing an ellipse in
        the probability simplex, or to `ellipse-clr` for constructing an
        ellipse in the Centered-Log Ratio (CLR) unconstrained space.
    :param bonferroni: bool (default False), whether to apply Bonferroni
        correction when `region='intervals'`. This parameter has no effect
        for ellipse-based regions.
    :param temperature: temperature (>0) for posterior calibration
        (default 1.)
    :param prior: an array-like with the alpha parameters of a Dirichlet
        prior, a scalar real value to be broadcast to all classes, or the
        string 'uniform' for a uniform, uninformative prior (default)
    :param verbose: bool, whether to display the progress bar
    """

    def __init__(
        self,
        classifier: BaseEstimator = None,
        fit_classifier=True,
        val_split: int = 5,
        kernel='gaussian',
        bandwidth=0.1,
        shrinkage=0.0,
        num_warmup: int = 500,
        num_samples: int = 1_000,
        mcmc_seed: int = 0,
        confidence_level: float = 0.95,
        region: str = 'intervals',
        bonferroni: bool = False,
        temperature: float = 1.0,
        prior='uniform',
        verbose: bool = False,
    ):
        _require_bayesian_dependencies()
        if num_warmup <= 0:
            raise ValueError(f'parameter {num_warmup=} must be a positive integer')
        if num_samples <= 0:
            raise ValueError(f'parameter {num_samples=} must be a positive integer')

        self.kernel = KDEBase._check_kernel(kernel)
        self.bandwidth = KDEBase._check_bandwidth(bandwidth, self.kernel)
        assert 0 <= shrinkage < 1, 'shrinkage must be in [0,1)'
        assert self.kernel != 'gaussian' or shrinkage == 0, \
            'shrinkage is only supported for Aitchison/ILR kernels'

        super().__init__(classifier, fit_classifier, val_split)
        self.shrinkage = float(shrinkage)
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.mcmc_seed = mcmc_seed
        self.confidence_level = confidence_level
        self.region = region
        self.bonferroni = bonferroni
        self.temperature = _validate_temperature(temperature)
        self.prior = prior
        self.verbose = verbose
        self.prevalence_samples = None

    def aggregation_fit(self, classif_predictions, labels):
        self.mix_densities = self.get_mixture_components(
            classif_predictions,
            labels,
            self.classes_,
            self.bandwidth,
            self.kernel,
        )
        return self

    def sample_from_posterior(self, classif_predictions):
        test_log_densities = np.asarray(
            [self.pdf(kde_i, classif_predictions, self.kernel, log_densities=True) for kde_i in self.mix_densities]
        )
        alpha = _resolve_dirichlet_prior(self.prior, len(self.mix_densities))

        mcmc = MCMC(
            NUTS(self._numpyro_model),
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=1,
            progress_bar=self.verbose,
        )
        mcmc.run(jrandom.PRNGKey(self.mcmc_seed), test_log_densities=test_log_densities, alpha=alpha)
        self.prevalence_samples = np.asarray(mcmc.get_samples()["prev"])
        return self.prevalence_samples

    def aggregate(self, classif_predictions: np.ndarray):
        return self.sample_from_posterior(classif_predictions).mean(axis=0)

    def predict_conf(self, instances, confidence_level=None) -> (np.ndarray, ConfidenceRegionABC):
        confidence_level = self.confidence_level if confidence_level is None else confidence_level
        classif_predictions = self.classify(instances)
        point_estimate = self.aggregate(classif_predictions)
        region = WithConfidenceABC.construct_region(
            self.prevalence_samples,
            confidence_level=confidence_level,
            method=self.region,
            bonferroni=self.bonferroni,
        )
        return point_estimate, region

    def _numpyro_model(self, test_log_densities, alpha):
        prev = numpyro.sample("prev", dist.Dirichlet(jnp.asarray(alpha)))
        log_likelihood = jnp.sum(jax_logsumexp(jnp.log(prev)[:, None] + test_log_densities, axis=0))
        numpyro.factor("loglik", (1.0 / self.temperature) * log_likelihood)


class _JaxILRTransformation(F.CompositionalTransformation):
    """
    JAX-backed ILR transform used inside Bayesian MAPLS.
    """

    def __call__(self, X):
        X = jnp.asarray(X)
        X = qp.error.smooth(np.asarray(X), self.EPSILON)
        X = jnp.asarray(X)
        basis = jnp.asarray(self.get_V(X.shape[-1]))
        return jnp.log(X) @ basis.T

    def inverse(self, Z):
        Z = jnp.asarray(Z)
        basis = jnp.asarray(self.get_V(Z.shape[-1] + 1))
        logp = Z @ basis
        p = jnp.exp(logp)
        return p / jnp.sum(p, axis=-1, keepdims=True)

    def get_V(self, k):
        return F.ILRtransformation().get_V(k)


class BayesianMAPLS(AggregativeSoftQuantifier, WithConfidenceABC):
    """
    Bayesian variant of the MLLS/EMQ method proposed by
    Ye, Changkun, et al. "Label shift estimation for class-imbalance problem:
    A bayesian approach." Proceedings of the IEEE/CVF Winter Conference on
    Applications of Computer Vision. 2024.

    Code adapted from:
    https://github.com/ChangkunYe/MAPLS/blob/main/label_shift/mapls.py

    This method relies on extra dependencies, which have to be installed via:
    `$ pip install quapy[bayes]`

    :param classifier: a scikit-learn's BaseEstimator, or None, in which case
        the classifier is taken to be the one indicated in
        `qp.environ['DEFAULT_CLS']`
    :param fit_classifier: whether to train the classifier, or consider it
        already fit
    :param val_split: specifies the data used for generating classifier
        predictions. This specification can be made as float in (0, 1)
        indicating the proportion of stratified held-out validation set to be
        extracted from the training set; or as an integer (default 5),
        indicating that the predictions are to be generated in a `k`-fold
        cross-validation manner (with this integer indicating the value for
        `k`); or as a tuple `(X,y)` defining the specific set of data to use
        for validation. Set to None when the method does not require any
        validation data, in order to avoid that some portion of the training
        data be wasted.
    :param exact_train_prev: set to True (default) for using the true training
        prevalence as the initial observation; set to False for computing the
        training prevalence as an estimate of it, i.e., as the expected value
        of the posterior probabilities of the training instances.
    :param num_warmup: number of warmup iterations for the MCMC sampler
        (default 500)
    :param num_samples: number of samples to draw from the posterior
        (default 1000)
    :param mcmc_seed: random seed for the MCMC sampler (default 0)
    :param confidence_level: float in [0,1] to construct a confidence region
        around the point estimate (default 0.95)
    :param region: string, set to `intervals` for constructing confidence
        intervals (default), or to `ellipse` for constructing an ellipse in
        the probability simplex, or to `ellipse-clr` for constructing an
        ellipse in the Centered-Log Ratio (CLR) unconstrained space.
    :param bonferroni: bool (default False), whether to apply Bonferroni
        correction when `region='intervals'`. This parameter has no effect
        for ellipse-based regions.
    :param temperature: temperature (>0) for posterior calibration
        (default 1.)
    :param prior: an array-like with the alpha parameters of a Dirichlet
        prior, a scalar real value to be broadcast to all classes, or one of
        {'uniform', 'map', 'map2'} (default 'uniform')
    :param mapls_chain_init: whether to initialize the Markov chain with a
        preliminary EM point estimate (default True)
    :param verbose: bool, whether to display the progress bar
        (default False)
    """

    def __init__(
        self,
        classifier: BaseEstimator = None,
        fit_classifier=True,
        val_split: int = 5,
        exact_train_prev=True,
        num_warmup: int = 500,
        num_samples: int = 1_000,
        mcmc_seed: int = 0,
        confidence_level: float = 0.95,
        region: str = 'intervals',
        bonferroni: bool = False,
        temperature: float = 1.0,
        prior='uniform',
        mapls_chain_init=True,
        verbose=False,
    ):
        _require_bayesian_dependencies()
        if num_warmup <= 0:
            raise ValueError(f'parameter {num_warmup=} must be a positive integer')
        if num_samples <= 0:
            raise ValueError(f'parameter {num_samples=} must be a positive integer')
        if not (
            (isinstance(prior, str) and prior in {'uniform', 'map', 'map2'})
            or isinstance(prior, Number)
            or (isinstance(prior, Iterable) and all(isinstance(v, Number) for v in prior))
        ):
            raise ValueError(
                f'wrong type for {prior=}; expected one of {{"uniform", "map", "map2"}}, '
                'a real scalar, or an array-like of real values'
            )

        super().__init__(classifier, fit_classifier, val_split)
        self.exact_train_prev = exact_train_prev
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.mcmc_seed = mcmc_seed
        self.confidence_level = confidence_level
        self.region = region
        self.bonferroni = bonferroni
        self.temperature = _validate_temperature(temperature)
        self.prior = prior
        self.mapls_chain_init = mapls_chain_init
        self.verbose = verbose
        self.prevalence_samples = None

    def aggregation_fit(self, classif_predictions, labels):
        self.train_post = classif_predictions
        if self.exact_train_prev:
            self.train_prevalence = F.prevalence_from_labels(labels, classes=self.classes_)
        else:
            self.train_prevalence = F.prevalence_from_probabilities(classif_predictions)
        self.ilr = _JaxILRTransformation()
        return self

    def sample_from_posterior(self, classif_predictions):
        n_test, n_classes = classif_predictions.shape
        map_estimate, lam = mapls(
            self.train_post,
            test_probs=classif_predictions,
            pz=self.train_prevalence,
            return_lambda=True,
        )

        z0 = self.ilr(map_estimate)
        if isinstance(self.prior, str) and self.prior in {'map', 'map2'}:
            prior_context = {
                "test_probs": classif_predictions,
                "train_prev": self.train_prevalence,
                "map_estimate": map_estimate,
            }
            alpha = _resolve_dirichlet_prior(
                self.prior,
                n_classes,
                allow_mapls_priors=True,
                n_test=n_test,
                map_prev=prior_context,
                map_lambda=lam,
            )
        else:
            alpha = _resolve_dirichlet_prior(self.prior, n_classes)

        mcmc = MCMC(
            NUTS(self._numpyro_model),
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=1,
            progress_bar=self.verbose,
        )
        mcmc.run(
            jrandom.PRNGKey(self.mcmc_seed),
            test_posteriors=classif_predictions,
            alpha=alpha,
            init_params={"z": z0} if self.mapls_chain_init else None,
        )

        samples = mcmc.get_samples()["z"]
        self.prevalence_samples = np.asarray(self.ilr.inverse(samples))
        return self.prevalence_samples

    def aggregate(self, classif_predictions: np.ndarray):
        return self.sample_from_posterior(classif_predictions).mean(axis=0)

    def predict_conf(self, instances, confidence_level=None) -> (np.ndarray, ConfidenceRegionABC):
        confidence_level = self.confidence_level if confidence_level is None else confidence_level
        classif_predictions = self.classify(instances)
        point_estimate = self.aggregate(classif_predictions)
        region = WithConfidenceABC.construct_region(
            self.prevalence_samples,
            confidence_level=confidence_level,
            method=self.region,
            bonferroni=self.bonferroni,
        )
        return point_estimate, region

    def _log_likelihood(self, test_classif, test_prev, train_prev):
        log_w = jnp.log(test_prev) - jnp.log(train_prev)
        return jnp.sum(jax_logsumexp(jnp.log(test_classif) + log_w, axis=-1))

    def _numpyro_model(self, test_posteriors, alpha):
        test_posteriors = jnp.asarray(test_posteriors)
        n_classes = test_posteriors.shape[1]

        z = numpyro.sample("z", dist.Normal(jnp.zeros(n_classes - 1), 1.0))
        prev = self.ilr.inverse(z)
        train_prev = jnp.asarray(self.train_prevalence)
        alpha = jnp.asarray(alpha)

        numpyro.factor("dirichlet_prior", dist.Dirichlet(alpha).log_prob(prev))
        numpyro.factor(
            "likelihood",
            (1.0 / self.temperature) * self._log_likelihood(test_posteriors, test_prev=prev, train_prev=train_prev),
        )


def mapls(
    train_probs: np.ndarray,
    test_probs: np.ndarray,
    pz: np.ndarray,
    qy_mode: str = 'soft',
    max_iter: int = 100,
    init_mode: str = 'identical',
    lam: float = None,
    dvg_name='kl',
    return_lambda=False,
):
    cls_num = len(pz)
    assert test_probs.shape[-1] == cls_num
    if not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError(f'expected a non-negative integer for max_iter; found {max_iter!r}')

    if dvg_name == 'kl':
        dvg = kl_div
    elif dvg_name == 'js':
        dvg = js_div
    else:
        raise ValueError(f'Unsupported distribution distance measure {dvg_name!r}; expected "kl" or "js"')

    q_prior = np.ones(cls_num) / cls_num
    if lam is None:
        lam = get_lambda(test_probs, pz, q_prior, dvg=dvg, max_iter=max_iter)

    qz = mapls_em(
        test_probs,
        pz,
        lam,
        q_prior,
        cls_num,
        init_mode=init_mode,
        max_iter=max_iter,
        qy_mode=qy_mode,
    )
    return (qz, lam) if return_lambda else qz


def mapls_em(probs, pz, lam, q_prior, cls_num, init_mode='identical', max_iter=100, qy_mode='soft'):
    pz = np.asarray(pz, dtype=float)
    pz = pz / np.sum(pz)
    if init_mode == 'uniform':
        qz = np.ones(cls_num) / cls_num
    elif init_mode == 'identical':
        qz = pz.copy()
    else:
        raise ValueError('init_mode should be either "uniform" or "identical"')

    w = qz / pz
    for _ in range(max_iter):
        mapls_probs = normalized(probs * w, axis=-1, order=1)
        if qy_mode == 'hard':
            pred = np.argmax(mapls_probs, axis=-1)
            qz_new = np.bincount(pred.reshape(-1), minlength=cls_num)
        elif qy_mode == 'soft':
            qz_new = np.mean(mapls_probs, axis=0)
        else:
            raise ValueError('qy_mode should be either "soft" or "hard"')

        qz = lam * qz_new + (1 - lam) * q_prior
        qz /= qz.sum()
        w = qz / pz

    return qz


def get_lambda(test_probs, pz, q_prior, dvg, max_iter=50):
    n_classes = len(pz)
    qz_pred = mapls_em(test_probs, pz, 1, 0, n_classes, max_iter=max_iter)

    tu_div = dvg(qz_pred, q_prior)
    ts_div = dvg(qz_pred, pz)
    su_div = dvg(pz, q_prior)

    su_conf = 1 - lambda_forward(su_div, lambda_inverse(dpq=0.5, lam=0.2))
    tu_conf = lambda_forward(tu_div, lambda_inverse(dpq=0.5, lam=su_conf))
    ts_conf = lambda_forward(ts_div, lambda_inverse(dpq=0.5, lam=su_conf))

    confs = np.array([tu_conf, 1 - ts_conf])
    weights = np.array([0.9, 0.1])
    return np.sum(weights * confs)


def lambda_inverse(dpq, lam):
    return (1 / (1 - lam) - 1) / dpq


def lambda_forward(dpq, gamma):
    return gamma * dpq / (1 + gamma * dpq)


def get_lamda(test_probs, pz, q_prior, dvg, max_iter=50):
    return get_lambda(test_probs, pz, q_prior, dvg, max_iter=max_iter)


def lam_inv(dpq, lam):
    return lambda_inverse(dpq, lam)


def lam_forward(dpq, gamma):
    return lambda_forward(dpq, gamma)


def kl_div(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / (q[mask] + eps)))


def js_div(p, q):
    assert (np.abs(np.sum(p) - 1) < 1e-6) and (np.abs(np.sum(q) - 1) < 1e-6)
    m = (p + q) / 2
    return kl_div(p, m) / 2 + kl_div(q, m) / 2


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def alpha0_from_lambda(lam, n_test, n_classes):
    return 1 + n_test * (1 - lam) / (lam * n_classes)


def alpha0_from_lamda(lam, n_test, n_classes):
    return alpha0_from_lambda(lam, n_test, n_classes)


def calibrate_temperature(
    method: WithConfidenceABC,
    train: LabelledCollection,
    val_prot: AbstractProtocol,
    temp_grid=(0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 100.0),
    nominal_coverage: float = 0.95,
    amplitude_threshold=1.0,
    criterion: str = 'winkler',
    n_jobs: int = 1,
    verbose: bool = True,
):
    """
    Calibrates the temperature parameter of a Bayesian quantifier with
    confidence regions by selecting the value that yields the best validation
    trade-off between nominal coverage and region sharpness.

    The method is first fitted on ``train``. For each candidate temperature,
    the fitted quantifier is deep-copied, its ``temperature`` attribute is
    replaced, and it is evaluated on the samples generated by ``val_prot``.
    Candidate temperatures whose average region amplitude exceeds
    ``amplitude_threshold`` are discarded.

    When ``criterion='winkler'``, the surviving candidate with minimum mean
    Winkler score is selected. When ``criterion='auto'``, the selected
    temperature is the one whose empirical coverage is closest to
    ``nominal_coverage``.

    :param method: a quantifier implementing :class:`WithConfidenceABC` and
        exposing a writable ``temperature`` attribute
    :param train: training set used to fit the quantifier
    :param val_prot: validation protocol yielding pairs ``(sample, true_prev)``
    :param temp_grid: candidate temperatures to evaluate
    :param nominal_coverage: target confidence level used by the Winkler score
        and coverage selection
    :param amplitude_threshold: maximum allowed average simplex proportion of
        the region. It can also be set to ``'auto'`` to use a heuristic based
        on the number of classes
    :param criterion: either ``'winkler'`` (default) or ``'auto'``
    :param n_jobs: number of parallel jobs across candidate temperatures
    :param verbose: whether to display progress information
    :return: the selected temperature value
    """
    if not hasattr(method, 'temperature'):
        raise ValueError(f'{method.__class__.__name__} does not expose a temperature attribute')
    if not isinstance(method, WithConfidenceABC):
        raise TypeError(f'{method.__class__.__name__} is not an instance of WithConfidenceABC')
    if not 0 < nominal_coverage < 1:
        raise ValueError(f'{nominal_coverage=} must be in the interval (0,1)')
    if criterion not in {'auto', 'winkler'}:
        raise ValueError(f'unknown {criterion=}; valid ones are "auto" or "winkler"')
    if amplitude_threshold != 'auto':
        if not isinstance(amplitude_threshold, Real) or amplitude_threshold > 1.0:
            raise ValueError(
                f'wrong value for {amplitude_threshold=}; it must either be "auto" or a real value <= 1.0'
            )
    temperatures = sorted(_validate_temperature(temp) for temp in temp_grid)

    if amplitude_threshold == 'auto':
        amplitude_threshold = 0.1 / np.log(train.n_classes + 1)

    if amplitude_threshold > 0.1:
        print(f'warning: the {amplitude_threshold=} is too large; this may lead to uninformative regions')

    def _evaluate_temperature_job(job_id, temperature):
        local_method = copy.deepcopy(method)
        local_method.temperature = temperature

        coverage = 0
        amplitudes = []
        winklers = []
        errors = []

        pbar = tqdm(
            enumerate(val_prot()),
            position=job_id,
            total=val_prot.total(),
            disable=not verbose,
        )

        for i, (sample, prev) in pbar:
            point_estim, conf_region = local_method.predict_conf(sample)

            if prev in conf_region:
                coverage += 1

            amplitudes.append(conf_region.montecarlo_proportion(n_trials=50_000))
            if criterion == 'winkler':
                winklers.append(conf_region.mean_winkler_score(true_prev=prev, alpha=1 - nominal_coverage))
            errors.append(qp.error.mae(prev, point_estim))

            description = (
                f'job={job_id} T={temperature}: '
                f'MAE={np.mean(errors):.6f} '
                f'coverage={coverage / (i + 1) * 100:.2f}% '
                f'amplitude={np.mean(amplitudes) * 100:.4f}% '
            )
            if criterion == 'winkler':
                description += f'winkler={np.mean(winklers):.4f}'
            pbar.set_description(description)

        mean_coverage = coverage / val_prot.total()
        mean_amplitude = float(np.mean(amplitudes))
        mean_winkler = float(np.mean(winklers)) if criterion == 'winkler' else None
        mean_error = float(np.mean(errors))
        return temperature, mean_coverage, mean_amplitude, mean_winkler, mean_error

    method.fit(*train.Xy)
    raw_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_evaluate_temperature_job)(job_id, temperature)
        for job_id, temperature in tqdm(enumerate(temperatures), disable=not verbose)
    )
    filtered_results = [
        (temperature, coverage, amplitude, winkler, error)
        for temperature, coverage, amplitude, winkler, error in raw_results
        if amplitude < amplitude_threshold
    ]

    chosen_temperature = 1.0
    chosen_coverage = chosen_amplitude = chosen_winkler = chosen_error = None

    if filtered_results:
        if criterion == 'winkler':
            chosen_temperature, chosen_coverage, chosen_amplitude, chosen_winkler, chosen_error = min(
                filtered_results, key=lambda item: item[3]
            )
        else:
            chosen_temperature, chosen_coverage, chosen_amplitude, chosen_winkler, chosen_error = min(
                filtered_results, key=lambda item: abs(item[1] - nominal_coverage)
            )

    if verbose and chosen_coverage is not None:
        message = (
            f'\nChosen_temperature={chosen_temperature:.2f} got '
            f'MAE={chosen_error:.6f} '
            f'coverage={chosen_coverage * 100:.2f}% '
            f'amplitude={chosen_amplitude * 100:.4f}% '
        )
        if criterion == 'winkler':
            message += f'winkler={chosen_winkler:.4f}'
        print(message)

    return chosen_temperature


def temp_calibration(*args, **kwargs):
    """
    Backward-compatible alias for :func:`calibrate_temperature`.
    """
    return calibrate_temperature(*args, **kwargs)
