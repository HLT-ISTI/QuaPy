import unittest
import quapy as qp
from quapy.data import LabelledCollection
from quapy.functional import strprev
from sklearn.linear_model import LogisticRegression
import numpy as np
from quapy.method.aggregative import PACC
import quapy.functional as F


class TestReplicability(unittest.TestCase):

    def test_prediction_replicability(self):

        dataset = qp.datasets.fetch_UCIBinaryDataset('yeast')

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

        self.assertEqual(str_prev1, str_prev2)


    def test_samping_replicability(self):

        def equal_collections(c1, c2, value=True):
            self.assertEqual(np.all(c1.X == c2.X), value)
            self.assertEqual(np.all(c1.y == c2.y), value)
            if value:
                self.assertEqual(np.all(c1.classes_ == c2.classes_), value)

        X = list(map(str, range(100)))
        y = np.random.randint(0, 2, 100)
        data = LabelledCollection(instances=X, labels=y)

        sample1 = data.sampling(50)
        sample2 = data.sampling(50)
        equal_collections(sample1, sample2, False)

        sample1 = data.sampling(50, random_state=0)
        sample2 = data.sampling(50, random_state=0)
        equal_collections(sample1, sample2, True)

        sample1 = data.sampling(50, *[0.7, 0.3], random_state=0)
        sample2 = data.sampling(50, *[0.7, 0.3], random_state=0)
        equal_collections(sample1, sample2, True)

        with qp.util.temp_seed(0):
            sample1 = data.sampling(50, *[0.7, 0.3])
        with qp.util.temp_seed(0):
            sample2 = data.sampling(50, *[0.7, 0.3])
        equal_collections(sample1, sample2, True)

        sample1 = data.sampling(50, *[0.7, 0.3], random_state=0)
        sample2 = data.sampling(50, *[0.7, 0.3], random_state=0)
        equal_collections(sample1, sample2, True)

        sample1_tr, sample1_te = data.split_stratified(train_prop=0.7, random_state=0)
        sample2_tr, sample2_te = data.split_stratified(train_prop=0.7, random_state=0)
        equal_collections(sample1_tr, sample2_tr, True)
        equal_collections(sample1_te, sample2_te, True)

        with qp.util.temp_seed(0):
            sample1_tr, sample1_te = data.split_stratified(train_prop=0.7)
        with qp.util.temp_seed(0):
            sample2_tr, sample2_te = data.split_stratified(train_prop=0.7)
        equal_collections(sample1_tr, sample2_tr, True)
        equal_collections(sample1_te, sample2_te, True)


    def test_parallel_replicability(self):

        train, test = qp.datasets.fetch_UCIMulticlassDataset('dry-bean').reduce().train_test

        test = test.sampling(500, *[0.1, 0.0, 0.1, 0.1, 0.2, 0.5, 0.0])

        with qp.util.temp_seed(10):
            pacc = PACC(LogisticRegression(), val_split=2, n_jobs=2)
            pacc.fit(train, val_split=0.5)
            prev1 = F.strprev(pacc.quantify(test.instances))

        with qp.util.temp_seed(0):
            pacc = PACC(LogisticRegression(), val_split=2, n_jobs=2)
            pacc.fit(train, val_split=0.5)
            prev2 = F.strprev(pacc.quantify(test.instances))

        with qp.util.temp_seed(0):
            pacc = PACC(LogisticRegression(), val_split=2, n_jobs=2)
            pacc.fit(train, val_split=0.5)
            prev3 = F.strprev(pacc.quantify(test.instances))

        print(prev1)
        print(prev2)
        print(prev3)

        self.assertNotEqual(prev1, prev2)
        self.assertEqual(prev2, prev3)




if __name__ == '__main__':
    unittest.main()
