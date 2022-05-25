import unittest
import quapy as qp
from sklearn.linear_model import LogisticRegression
from time import time
from method.aggregative import EMQ
from method.base import BaseQuantifier


class EvalTestCase(unittest.TestCase):
    def test_eval_speedup(self):

        data = qp.datasets.fetch_reviews('hp', tfidf=True, min_df=10, pickle=True)
        train, test = data.training, data.test

        protocol = qp.protocol.APP(test, sample_size=1000, n_prevalences=21, repeats=1, random_seed=1)

        class SlowLR(LogisticRegression):
            def predict_proba(self, X):
                import time
                time.sleep(1)
                return super().predict_proba(X)

        emq = EMQ(SlowLR()).fit(train)

        tinit = time()
        score = qp.evaluation.evaluate(emq, protocol, error_metric='mae', verbose=True)
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

            def set_params(self, **parameters): pass
            def get_params(self, deep=True): pass


        emq = NonAggregativeEMQ(SlowLR()).fit(train)

        tinit = time()
        score = qp.evaluation.evaluate(emq, protocol, error_metric='mae', verbose=True)
        tend_no_optim = time() - tinit
        print(f'evaluation (w/o optimization) took {tend_no_optim}s [MAE={score:.4f}]')

        self.assertEqual(tend_no_optim>tend_optim, True)


if __name__ == '__main__':
    unittest.main()
