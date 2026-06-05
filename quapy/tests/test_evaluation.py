import inspect
import unittest
from time import time

import numpy as np
from sklearn.linear_model import LogisticRegression

import quapy as qp
from quapy.error import QUANTIFICATION_ERROR_SINGLE_NAMES
from quapy.method.aggregative import EMQ, PCC
from quapy.method.base import BaseQuantifier
from quapy.tests._synthetic import make_dataset


class EvalTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = make_dataset(n_train=140, n_test=90, n_classes=2, random_state=7, name='eval')

    def test_eval_speedup(self):
        train, test = self.data.training, self.data.test
        protocol = qp.protocol.APP(test, sample_size=30, n_prevalences=5, repeats=1, random_state=1)

        class SlowLR(LogisticRegression):
            def predict_proba(self, X):
                import time as _time
                _time.sleep(0.05)
                return super().predict_proba(X)

        emq = EMQ(SlowLR(max_iter=1000)).fit(*train.Xy)

        tinit = time()
        score = qp.evaluation.evaluate(emq, protocol, error_metric='mae', aggr_speedup='force')
        tend_optim = time() - tinit
        self.assertTrue(isinstance(score, float))

        class NonAggregativeEMQ(BaseQuantifier):
            def __init__(self, cls):
                self.emq = EMQ(cls)

            def predict(self, X):
                return self.emq.predict(X)

            def fit(self, X, y):
                self.emq.fit(X, y)
                return self

        emq = NonAggregativeEMQ(SlowLR(max_iter=1000)).fit(*train.Xy)

        tinit = time()
        score = qp.evaluation.evaluate(emq, protocol, error_metric='mae')
        tend_no_optim = time() - tinit
        self.assertTrue(isinstance(score, float))
        self.assertGreater(tend_no_optim, tend_optim)

    def test_evaluation_output(self):
        train, test = self.data.training, self.data.test
        qp.environ['SAMPLE_SIZE'] = 30
        protocol = qp.protocol.APP(test, sample_size=30, n_prevalences=5, repeats=1, random_state=0)
        q = PCC(LogisticRegression(max_iter=1000)).fit(*train.Xy)

        def supports_evaluation(err):
            required = [
                p for p in inspect.signature(err).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is inspect._empty
            ]
            return len(required) <= 2

        single_errors = [
            e for e in QUANTIFICATION_ERROR_SINGLE_NAMES
            if supports_evaluation(qp.error.from_name(e))
        ]
        averaged_errors = ['m' + e for e in single_errors]
        single_errors = single_errors + [qp.error.from_name(e) for e in single_errors]
        averaged_errors = averaged_errors + [qp.error.from_name(e) for e in averaged_errors]
        for error_metric, averaged_error_metric in zip(single_errors, averaged_errors):
            score = qp.evaluation.evaluate(q, protocol, error_metric=averaged_error_metric)
            self.assertTrue(isinstance(score, float))

            scores = qp.evaluation.evaluate(q, protocol, error_metric=error_metric)
            self.assertTrue(isinstance(scores, np.ndarray))
            self.assertEqual(scores.mean(), score)


if __name__ == '__main__':
    unittest.main()
