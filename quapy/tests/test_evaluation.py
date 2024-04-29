import unittest

import numpy as np

import quapy as qp
from sklearn.linear_model import LogisticRegression
from time import time

from quapy.error import QUANTIFICATION_ERROR_SINGLE_NAMES
from quapy.method.aggregative import EMQ, PCC
from quapy.method.base import BaseQuantifier


class EvalTestCase(unittest.TestCase):

    def test_eval_speedup(self):
        """
        Checks whether the speed-up heuristics used by qp.evaluation work, i.e., actually save time
        """

        data = qp.datasets.fetch_reviews('hp', tfidf=True, min_df=10, pickle=True)
        train, test = data.training, data.test

        protocol = qp.protocol.APP(test, sample_size=1000, n_prevalences=11, repeats=1, random_state=1)

        class SlowLR(LogisticRegression):
            def predict_proba(self, X):
                import time
                time.sleep(1)
                return super().predict_proba(X)

        emq = EMQ(SlowLR()).fit(train)

        tinit = time()
        score = qp.evaluation.evaluate(emq, protocol, error_metric='mae', verbose=True, aggr_speedup='force')
        tend_optim = time()-tinit
        print(f'evaluation (with optimization) took {tend_optim}s [MAE={score:.4f}]')

        class NonAggregativeEMQ(BaseQuantifier):

            def __init__(self, cls):
                self.emq = EMQ(cls)

            def quantify(self, instances):
                return self.emq.quantify(instances)

            def fit(self, data):
                self.emq.fit(data)
                return self

        emq = NonAggregativeEMQ(SlowLR()).fit(train)

        tinit = time()
        score = qp.evaluation.evaluate(emq, protocol, error_metric='mae', verbose=True)
        tend_no_optim = time() - tinit
        print(f'evaluation (w/o optimization) took {tend_no_optim}s [MAE={score:.4f}]')

        self.assertEqual(tend_no_optim>(tend_optim/2), True)

    def test_evaluation_output(self):
        """
        Checks the evaluation functions return correct types for different error_metrics
        """

        data = qp.datasets.fetch_reviews('hp', tfidf=True, min_df=10, pickle=True).reduce(n_train=100, n_test=100)
        train, test = data.training, data.test

        qp.environ['SAMPLE_SIZE']=100

        protocol = qp.protocol.APP(test, random_state=0)

        q = PCC(LogisticRegression()).fit(train)

        single_errors = list(QUANTIFICATION_ERROR_SINGLE_NAMES)
        averaged_errors = ['m'+e for e in single_errors]
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
