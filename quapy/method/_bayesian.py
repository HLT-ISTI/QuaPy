"""
Utility functions for `Bayesian quantification <https://arxiv.org/abs/2302.09159>`_ methods.
"""
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    DEPENDENCIES_INSTALLED = True
except ImportError:
    jax = None
    jnp = None
    numpyro = None
    dist = None

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
