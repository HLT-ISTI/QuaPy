import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from quapy.method.aggregative import PACC
from quapy.method.confidence import ConfidenceIntervals, ConfidenceEllipseSimplex, AggregativeBootstrap, WithConfidenceABC
from quapy.tests._synthetic import make_dataset


def _dirichlet_samples(n_classes=3, n_samples=300, random_state=0):
    rng = np.random.RandomState(random_state)
    return rng.dirichlet(np.ones(n_classes) * 5, size=n_samples)


class TestConfidenceRegions(unittest.TestCase):

    def test_confidence_intervals_contain_own_mean(self):
        samples = _dirichlet_samples()
        region = ConfidenceIntervals(samples)
        point_estimate = region.point_estimate()
        self.assertEqual(region.coverage(point_estimate), 1.)
        self.assertEqual(region.n_dim, 3)

    def test_confidence_ellipse_simplex_contains_own_mean(self):
        samples = _dirichlet_samples()
        region = ConfidenceEllipseSimplex(samples)
        point_estimate = region.point_estimate()
        self.assertIn(point_estimate, region)

    def test_construct_region_applies_bonferroni_only_to_intervals(self):
        samples = _dirichlet_samples()
        region_plain = WithConfidenceABC.construct_region(samples, confidence_level=0.9, method='intervals')
        region_bonf = WithConfidenceABC.construct_region(samples, confidence_level=0.9, method='intervals', bonferroni=True)
        ellipse = WithConfidenceABC.construct_region(samples, confidence_level=0.9, method='ellipse', bonferroni=True)

        self.assertAlmostEqual(region_plain.alpha, 0.1)
        self.assertAlmostEqual(region_bonf.alpha, 0.1 / samples.shape[1])
        self.assertIsInstance(ellipse, ConfidenceEllipseSimplex)

    def test_simplex_portion_is_cached_and_consistent(self):
        # regression test for the @lru_cache-on-bound-method memory leak fix:
        # results must still be memoized per instance, and two instances must not share state
        region1 = ConfidenceEllipseSimplex(_dirichlet_samples(random_state=1))
        region2 = ConfidenceEllipseSimplex(_dirichlet_samples(random_state=2))

        p1_first = region1.simplex_portion()
        p1_second = region1.simplex_portion()
        self.assertEqual(p1_first, p1_second)

        p2 = region2.simplex_portion()
        self.assertTrue(hasattr(region1, '_simplex_portion_cache'))
        self.assertTrue(hasattr(region2, '_simplex_portion_cache'))
        # each instance keeps its own cached value
        self.assertEqual(region1._simplex_portion_cache, p1_first)
        self.assertEqual(region2._simplex_portion_cache, p2)

    def test_aggregative_bootstrap_end_to_end(self):
        dataset = make_dataset(n_train=150, n_test=50, n_classes=3, n_features=12, random_state=5)
        learner = LogisticRegression(max_iter=2000)
        learner.fit(*dataset.training.Xy)
        quantifier = AggregativeBootstrap(
            PACC(learner, fit_classifier=False), n_test_samples=50, confidence_level=0.9
        )
        quantifier.fit(*dataset.training.Xy)
        point_estimate, region = quantifier.predict_conf(dataset.test.X)
        self.assertEqual(len(point_estimate), 3)
        self.assertEqual(region.coverage(point_estimate), 1.)

    def test_aggregative_bootstrap_exposes_bonferroni(self):
        dataset = make_dataset(n_train=150, n_test=50, n_classes=3, n_features=12, random_state=7)
        learner = LogisticRegression(max_iter=2000)
        learner.fit(*dataset.training.Xy)
        quantifier = AggregativeBootstrap(
            PACC(learner, fit_classifier=False),
            n_test_samples=20,
            confidence_level=0.9,
            region='intervals',
            bonferroni=True,
            random_state=0,
        )
        quantifier.fit(*dataset.training.Xy)
        _, region = quantifier.predict_conf(dataset.test.X)

        self.assertAlmostEqual(region.alpha, 0.1 / 3)


if __name__ == '__main__':
    unittest.main()
