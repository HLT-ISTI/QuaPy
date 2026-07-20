import unittest

import numpy as np

import quapy.functional as F


class TestFunctional(unittest.TestCase):

    def test_ternary_search_binary(self):
        def loss(prev):
            return (prev[1] - 0.37) ** 2

        result = F.argmin_prevalence(loss, n_classes=2, method='ternary_search')
        self.assertTrue(np.allclose(result.sum(), 1.0))
        self.assertAlmostEqual(result[1], 0.37, places=3)

    def test_ternary_search_multiclass_not_supported(self):
        def loss(prev):
            return np.sum((prev - np.array([0.2, 0.3, 0.5])) ** 2)

        with self.assertRaises(AssertionError):
            F.argmin_prevalence(loss, n_classes=3, method='ternary_search')


if __name__ == '__main__':
    unittest.main()
