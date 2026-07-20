"""
Integration tests for optional or resource-heavy method end-to-end checks.

This module is intentionally excluded from default ``unittest`` discovery by
using an ``integration_*.py`` filename.
"""

import unittest

import quapy as qp
from quapy.functional import check_prevalence_vector


class IntegrationMethodsTest(unittest.TestCase):

    def test_quanet(self):
        try:
            import quapy.classification.neural
        except ModuleNotFoundError:
            print('the torch package is not installed; skipping integration test for QuaNet')
            return

        qp.environ['SAMPLE_SIZE'] = 10

        dataset = qp.datasets.fetch_reviews('kindle', pickle=True).reduce()
        qp.data.preprocessing.index(dataset, min_df=5, inplace=True)

        from quapy.classification.neural import CNNnet
        from quapy.classification.neural import NeuralClassifierTrainer
        from quapy.method.meta import QuaNet

        cnn = CNNnet(dataset.vocabulary_size, dataset.n_classes)
        learner = NeuralClassifierTrainer(cnn, device='cpu')
        model = QuaNet(learner, device='cpu', n_epochs=2, tr_iter_per_poch=10, va_iter_per_poch=10, patience=2)

        model.fit(*dataset.training.Xy)
        estim_prevalences = model.predict(dataset.test.instances)
        self.assertTrue(check_prevalence_vector(estim_prevalences))


if __name__ == '__main__':
    unittest.main()
