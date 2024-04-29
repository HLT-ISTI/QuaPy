import itertools
import unittest

from sklearn.linear_model import LogisticRegression

import quapy as qp
from quapy.method.aggregative import ACC
from quapy.method.meta import Ensemble
from quapy.method import AGGREGATIVE_METHODS, BINARY_METHODS, NON_AGGREGATIVE_METHODS
from quapy.functional import check_prevalence_vector

# a random selection of composed methods to test the qunfold integration
from quapy.method.composable import (
    ComposableQuantifier,
    LeastSquaresLoss,
    HellingerSurrogateLoss,
    ClassTransformer,
    HistogramTransformer,
    CVClassifier,
)
COMPOSABLE_METHODS = [
    ComposableQuantifier( # ACC
        LeastSquaresLoss(),
        ClassTransformer(CVClassifier(LogisticRegression()))
    ),
    ComposableQuantifier( # HDy
        HellingerSurrogateLoss(),
        HistogramTransformer(
            3, # 3 bins per class
            preprocessor = ClassTransformer(CVClassifier(LogisticRegression()))
        )
    ),
]

class TestMethods(unittest.TestCase):

    tiny_dataset_multiclass = qp.datasets.fetch_UCIMulticlassDataset('academic-success').reduce(n_test=10)
    tiny_dataset_binary = qp.datasets.fetch_UCIBinaryDataset('ionosphere').reduce(n_test=10)
    datasets = [tiny_dataset_binary, tiny_dataset_multiclass]

    def test_aggregative(self):
        for dataset in TestMethods.datasets:
            learner = LogisticRegression()
            learner.fit(*dataset.training.Xy)

            for model in AGGREGATIVE_METHODS:
                if not dataset.binary and model in BINARY_METHODS:
                    print(f'skipping the test of binary model {model.__name__} on multiclass dataset {dataset.name}')
                    continue

                q = model(learner)
                print('testing', q)
                q.fit(dataset.training, fit_classifier=False)
                estim_prevalences = q.quantify(dataset.test.X)
                self.assertTrue(check_prevalence_vector(estim_prevalences))

    def test_non_aggregative(self):
        for dataset in TestMethods.datasets:

            for model in NON_AGGREGATIVE_METHODS:
                if not dataset.binary and model in BINARY_METHODS:
                    print(f'skipping the test of binary model {model.__name__} on multiclass dataset {dataset.name}')
                    continue

                q = model()
                print(f'testing {q} on dataset {dataset.name}')
                q.fit(dataset.training)
                estim_prevalences = q.quantify(dataset.test.X)
                self.assertTrue(check_prevalence_vector(estim_prevalences))

    def test_ensembles(self):

        qp.environ['SAMPLE_SIZE'] = 10

        base_quantifier = ACC(LogisticRegression())
        for dataset, policy in itertools.product(TestMethods.datasets, Ensemble.VALID_POLICIES):
            if not dataset.binary and policy == 'ds':
                print(f'skipping the test of binary policy ds on non-binary dataset {dataset}')
                continue

            print(f'testing {base_quantifier} on dataset {dataset.name} with {policy=}')
            ensemble = Ensemble(quantifier=base_quantifier, size=3, policy=policy, n_jobs=-1)
            ensemble.fit(dataset.training)
            estim_prevalences = ensemble.quantify(dataset.test.instances)
            self.assertTrue(check_prevalence_vector(estim_prevalences))

    def test_quanet(self):
        try:
            import quapy.classification.neural
        except ModuleNotFoundError:
            print('the torch package is not installed; skipping unit test for QuaNet')
            return

        qp.environ['SAMPLE_SIZE'] = 10

        # load the kindle dataset as text, and convert words to numerical indexes
        dataset = qp.datasets.fetch_reviews('kindle', pickle=True).reduce()
        qp.data.preprocessing.index(dataset, min_df=5, inplace=True)

        from quapy.classification.neural import CNNnet
        cnn = CNNnet(dataset.vocabulary_size, dataset.n_classes)

        from quapy.classification.neural import NeuralClassifierTrainer
        learner = NeuralClassifierTrainer(cnn, device='cpu')

        from quapy.method.meta import QuaNet
        model = QuaNet(learner, device='cpu', n_epochs=2, tr_iter_per_poch=10, va_iter_per_poch=10, patience=2)

        model.fit(dataset.training)
        estim_prevalences = model.quantify(dataset.test.instances)
        self.assertTrue(check_prevalence_vector(estim_prevalences))

    def test_composable(self):
        for dataset in TestMethods.datasets:
            for q in COMPOSABLE_METHODS:
                print('testing', q)
                q.fit(dataset.training)
                estim_prevalences = q.quantify(dataset.test.X)
                self.assertTrue(check_prevalence_vector(estim_prevalences))


if __name__ == '__main__':
    unittest.main()
