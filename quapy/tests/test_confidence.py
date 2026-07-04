import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from quapy.method.aggregative import PACC
from quapy.method.confidence import ConfidenceIntervals, ConfidenceEllipseSimplex, AggregativeBootstrap
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


if __name__ == '__main__':
    unittest.main()
