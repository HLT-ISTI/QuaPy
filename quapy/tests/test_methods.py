import itertools
import inspect
import unittest

from sklearn.linear_model import LogisticRegression

from quapy.method import AGGREGATIVE_METHODS, BINARY_METHODS, NON_AGGREGATIVE_METHODS
from quapy.method.aggregative import ACC
from quapy.method.meta import Ensemble
from quapy.functional import check_prevalence_vector
from quapy.tests._synthetic import make_dataset
import quapy as qp

OPTIONAL_AGGREGATIVE_METHODS = {
    'BayesianCC',
    'BayesianKDEy',
    'BayesianMAPLS',
    'PQ',
}


class TestMethods(unittest.TestCase):

    tiny_dataset_multiclass = make_dataset(
        n_train=140, n_test=40, n_classes=3, n_features=12, random_state=11, name='synthetic-multiclass'
    )
    tiny_dataset_binary = make_dataset(
        n_train=140, n_test=40, n_classes=2, n_features=12, random_state=13, name='synthetic-binary'
    )
    datasets = [tiny_dataset_binary, tiny_dataset_multiclass]

    def test_aggregative(self):
        for dataset in TestMethods.datasets:
            learner = LogisticRegression(max_iter=2000)
            learner.fit(*dataset.training.Xy)

            for model in AGGREGATIVE_METHODS:
                if model.__name__ in OPTIONAL_AGGREGATIVE_METHODS:
                    continue
                if not dataset.binary and model in BINARY_METHODS:
                    continue

                kwargs = {'fit_classifier': False}
                if 'val_split' in inspect.signature(model.__init__).parameters:
                    kwargs['val_split'] = None
                q = model(learner, **kwargs)
                q.fit(*dataset.training.Xy)
                estim_prevalences = q.predict(dataset.test.X)
                self.assertTrue(check_prevalence_vector(estim_prevalences))

    def test_non_aggregative(self):
        for dataset in TestMethods.datasets:
            for model in NON_AGGREGATIVE_METHODS:
                if not dataset.binary and model in BINARY_METHODS:
                    continue

                q = model()
                q.fit(*dataset.training.Xy)
                estim_prevalences = q.predict(dataset.test.X)
                self.assertTrue(check_prevalence_vector(estim_prevalences))

    def test_ensembles(self):
        qp.environ['SAMPLE_SIZE'] = 20

        def policy_supported(policy):
            if policy in {'ave', 'ptr', 'ds'}:
                return True
            err = qp.error.from_name(policy)
            required = [
                p for p in inspect.signature(err).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is inspect._empty
            ]
            return len(required) <= 2

        base_quantifier = ACC(LogisticRegression(max_iter=2000))
        for dataset, policy in itertools.product(TestMethods.datasets, Ensemble.VALID_POLICIES):
            if not policy_supported(policy):
                continue
            if not dataset.binary and policy == 'ds':
                continue

            ensemble = Ensemble(quantifier=base_quantifier, size=3, policy=policy, n_jobs=1)
            ensemble.fit(*dataset.training.Xy)
            estim_prevalences = ensemble.predict(dataset.test.instances)
            self.assertTrue(check_prevalence_vector(estim_prevalences))

    def test_composable(self):
        try:
            from quapy.method.composable import check_compatible_qunfold_version
            from quapy.method.composable import (
                ComposableQuantifier,
                LeastSquaresLoss,
                HellingerSurrogateLoss,
                ClassRepresentation,
                HistogramRepresentation,
                CVClassifier,
            )
        except ImportError:
            return

        composable_methods = [
            ComposableQuantifier(
                LeastSquaresLoss(),
                ClassRepresentation(CVClassifier(LogisticRegression()))
            ),
            ComposableQuantifier(
                HellingerSurrogateLoss(),
                HistogramRepresentation(
                    3,
                    preprocessor=ClassRepresentation(CVClassifier(LogisticRegression()))
                )
            ),
        ]

        if check_compatible_qunfold_version():
            for dataset in TestMethods.datasets:
                for q in composable_methods:
                    q.fit(*dataset.training.Xy)
                    estim_prevalences = q.predict(dataset.test.X)
                    self.assertTrue(check_prevalence_vector(estim_prevalences))
        else:
            from quapy.method.composable import __old_version_message
            print(__old_version_message)


if __name__ == '__main__':
    unittest.main()
