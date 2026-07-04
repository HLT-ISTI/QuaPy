import unittest

import numpy as np

from quapy.method._bayesian import (
    _validate_temperature,
    _resolve_dirichlet_prior,
    kl_div,
    js_div,
    normalized,
    lambda_inverse,
    lambda_forward,
)


class TestBayesianUtils(unittest.TestCase):
    """
    Smoke tests for the pure-numpy helper functions in method/_bayesian.py. These do not require the
    optional jax/stan/numpyro dependencies (unlike BayesianKDEy/BayesianMAPLS themselves), so they can
    run regardless of whether `quapy[bayes]` is installed.
    """

    def test_validate_temperature(self):
        self.assertEqual(_validate_temperature(2), 2.0)
        self.assertEqual(_validate_temperature(0.5), 0.5)
        with self.assertRaises(ValueError):
            _validate_temperature(0)
        with self.assertRaises(ValueError):
            _validate_temperature(-1)
        with self.assertRaises(ValueError):
            _validate_temperature('not-a-number')

    def test_resolve_dirichlet_prior(self):
        np.testing.assert_array_equal(_resolve_dirichlet_prior('uniform', n_classes=3), np.ones(3))
        np.testing.assert_array_equal(_resolve_dirichlet_prior(2.5, n_classes=3), np.full(3, 2.5))
        np.testing.assert_array_equal(_resolve_dirichlet_prior([1, 2, 3], n_classes=3), np.array([1., 2., 3.]))
        with self.assertRaises(ValueError):
            _resolve_dirichlet_prior([1, 2], n_classes=3)  # wrong shape
        with self.assertRaises(ValueError):
            _resolve_dirichlet_prior('unknown-prior', n_classes=3)

    def test_kl_div_and_js_div(self):
        p = np.array([0.5, 0.5])
        self.assertAlmostEqual(kl_div(p, p), 0.0, places=6)
        self.assertAlmostEqual(js_div(p, p), 0.0, places=6)

        q = np.array([0.9, 0.1])
        self.assertGreater(kl_div(p, q), 0.0)
        self.assertGreater(js_div(p, q), 0.0)
        # JS divergence is symmetric, unlike KL
        self.assertAlmostEqual(js_div(p, q), js_div(q, p), places=6)

    def test_normalized(self):
        # normalized() operates on a batch of row vectors, shape (n_samples, n_features)
        a = np.array([[3.0, 4.0], [0.0, 0.0]])
        normed = normalized(a)
        self.assertAlmostEqual(np.linalg.norm(normed[0]), 1.0, places=6)
        # an all-zero row should not raise (guarded division), and stays all-zero
        np.testing.assert_array_equal(normed[1], np.array([0.0, 0.0]))

    def test_lambda_inverse_forward_roundtrip(self):
        dpq, lam = 0.3, 0.4
        gamma = lambda_inverse(dpq, lam)
        recovered_lam = lambda_forward(dpq, gamma)
        self.assertAlmostEqual(recovered_lam, lam, places=6)


if __name__ == '__main__':
    unittest.main()
