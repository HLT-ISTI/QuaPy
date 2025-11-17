"""
Utility functions for `Bayesian quantification <https://arxiv.org/abs/2302.09159>`_ methods.
"""
import numpy as np
import importlib.resources

try:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    import stan

    DEPENDENCIES_INSTALLED = True
except ImportError:
    jax = None
    jnp = None
    numpyro = None
    dist = None
    stan = None

    DEPENDENCIES_INSTALLED = False


P_TEST_Y: str = "P_test(Y)"
P_TEST_C: str = "P_test(C)"
P_C_COND_Y: str = "P(C|Y)"


def model(n_c_unlabeled: np.ndarray, n_y_and_c_labeled: np.ndarray) -> None:
    """
    Defines a probabilistic model in `NumPyro <https://num.pyro.ai/>`_.

    :param n_c_unlabeled: a `np.ndarray` of shape `(n_predicted_classes,)`
        with entry `c` being the number of instances predicted as class `c`.
    :param n_y_and_c_labeled: a `np.ndarray` of shape `(n_classes, n_predicted_classes)`
        with entry `(y, c)` being the number of instances labeled as class `y` and predicted as class `c`.
    """
    n_y_labeled = n_y_and_c_labeled.sum(axis=1)

    K = len(n_c_unlabeled)
    L = len(n_y_labeled)

    pi_ = numpyro.sample(P_TEST_Y, dist.Dirichlet(jnp.ones(L)))
    p_c_cond_y = numpyro.sample(P_C_COND_Y, dist.Dirichlet(jnp.ones(K).repeat(L).reshape(L, K)))

    with numpyro.plate('plate', L):
        numpyro.sample('F_yc', dist.Multinomial(n_y_labeled, p_c_cond_y), obs=n_y_and_c_labeled)

    p_c = numpyro.deterministic(P_TEST_C, jnp.einsum("yc,y->c", p_c_cond_y, pi_))
    numpyro.sample('N_c', dist.Multinomial(jnp.sum(n_c_unlabeled), p_c), obs=n_c_unlabeled)


def sample_posterior(
    n_c_unlabeled: np.ndarray,
    n_y_and_c_labeled: np.ndarray,
    num_warmup: int,
    num_samples: int,
    seed: int = 0,
) -> dict:
    """
    Samples from the Bayesian quantification model in NumPyro using the
    `NUTS <https://arxiv.org/abs/1111.4246>`_ sampler.

    :param n_c_unlabeled: a `np.ndarray` of shape `(n_predicted_classes,)`
        with entry `c` being the number of instances predicted as class `c`.
    :param n_y_and_c_labeled: a `np.ndarray` of shape `(n_classes, n_predicted_classes)`
        with entry `(y, c)` being the number of instances labeled as class `y` and predicted as class `c`.
    :param num_warmup: the number of warmup steps.
    :param num_samples: the number of samples to draw.
    :seed: the random seed.
    :return: a `dict` with the samples. The keys are the names of the latent variables.
    """
    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=False
    )
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, n_c_unlabeled=n_c_unlabeled, n_y_and_c_labeled=n_y_and_c_labeled)
    return mcmc.get_samples()



def load_stan_file():
    return importlib.resources.files('quapy.method').joinpath('stan/pq.stan').read_text(encoding='utf-8')

def pq_stan(stan_code, n_bins, pos_hist, neg_hist, test_hist, number_of_samples, num_warmup, stan_seed):
    """
    Perform Bayesian prevalence estimation using a Stan model for probabilistic quantification.

    This function builds and samples from a Stan model that implements a bin-based Bayesian
    quantifier. It uses the class-conditional histograms of the classifier
    outputs for positive and negative examples, along with the test histogram, to estimate
    the posterior distribution of prevalence in the test set.

    Parameters
    ----------
    stan_code : str
        The Stan model code as a string. 
    n_bins : int
        Number of bins used to build the histograms for positive and negative examples.
    pos_hist : array-like of shape (n_bins,)
        Histogram counts of the classifier outputs for the positive class.
    neg_hist : array-like of shape (n_bins,)
        Histogram counts of the classifier outputs for the negative class.
    test_hist : array-like of shape (n_bins,)
        Histogram counts of the classifier outputs for the test set, binned using the same bins.
    number_of_samples : int
        Number of post-warmup samples to draw from the Stan posterior.
    num_warmup : int
        Number of warmup iterations for the sampler.
    stan_seed : int
        Random seed for Stan model compilation and sampling, ensuring reproducibility.

    Returns
    -------
    prev_samples : numpy.ndarray
        An array of posterior samples of the prevalence (`prev`) in the test set.
        Each element corresponds to one draw from the posterior distribution.
    """

    stan_data = {
            'n_bucket': n_bins,
            'train_neg': neg_hist.tolist(),
            'train_pos': pos_hist.tolist(),
            'test': test_hist.tolist(),
            'posterior': 1
        }

    stan_model = stan.build(stan_code, data=stan_data, random_seed=stan_seed)
    fit = stan_model.sample(num_chains=1, num_samples=number_of_samples,num_warmup=num_warmup)

    return fit['prev']
