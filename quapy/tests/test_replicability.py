import unittest
import quapy as qp
from quapy.functional import strprev
from sklearn.linear_model import LogisticRegression

from method.aggregative import PACC


class MyTestCase(unittest.TestCase):
    def test_replicability(self):

        dataset = qp.datasets.fetch_UCIDataset('yeast')

        with qp.util.temp_seed(0):
            lr = LogisticRegression(random_state=0, max_iter=10000)
            pacc = PACC(lr)
            prev = pacc.fit(dataset.training).quantify(dataset.test.X)
            str_prev1 = strprev(prev, prec=5)

        with qp.util.temp_seed(0):
            lr = LogisticRegression(random_state=0, max_iter=10000)
            pacc = PACC(lr)
            prev2 = pacc.fit(dataset.training).quantify(dataset.test.X)
            str_prev2 = strprev(prev2, prec=5)

        self.assertEqual(str_prev1, str_prev2)  # add assertion here


if __name__ == '__main__':
    unittest.main()
