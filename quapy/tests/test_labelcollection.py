import unittest
import numpy as np
from scipy.sparse import csr_matrix

import quapy as qp


class LabelCollectionTestCase(unittest.TestCase):
    def test_split(self):
        x = np.arange(100)
        y = np.random.randint(0,5,100)
        data = qp.data.LabelledCollection(x,y)
        tr, te = data.split_random(0.7)
        check_prev = tr.prevalence()*0.7 + te.prevalence()*0.3

        self.assertEqual(len(tr), 70)
        self.assertEqual(len(te), 30)
        self.assertEqual(np.allclose(check_prev, data.prevalence()), True)
        self.assertEqual(len(tr+te), len(data))

    def test_join(self):
        x = np.arange(50)
        y = np.random.randint(2, 5, 50)
        data1 = qp.data.LabelledCollection(x, y)

        x = np.arange(200)
        y = np.random.randint(0, 3, 200)
        data2 = qp.data.LabelledCollection(x, y)

        x = np.arange(100)
        y = np.random.randint(0, 6, 100)
        data3 = qp.data.LabelledCollection(x, y)

        combined = qp.data.LabelledCollection.join(data1, data2, data3)
        self.assertEqual(len(combined), len(data1)+len(data2)+len(data3))
        self.assertEqual(all(combined.classes_ == np.arange(6)), True)

        x = np.random.rand(10, 3)
        y = np.random.randint(0, 1, 10)
        data4 = qp.data.LabelledCollection(x, y)
        with self.assertRaises(Exception):
            combined = qp.data.LabelledCollection.join(data1, data2, data3, data4)

        x = np.random.rand(20, 3)
        y = np.random.randint(0, 1, 20)
        data5 = qp.data.LabelledCollection(x, y)
        combined = qp.data.LabelledCollection.join(data4, data5)
        self.assertEqual(len(combined), len(data4)+len(data5))

        x = np.random.rand(10, 4)
        y = np.random.randint(0, 1, 10)
        data6 = qp.data.LabelledCollection(x, y)
        with self.assertRaises(Exception):
            combined = qp.data.LabelledCollection.join(data4, data5, data6)

        data4.instances = csr_matrix(data4.instances)
        with self.assertRaises(Exception):
            combined = qp.data.LabelledCollection.join(data4, data5)
        data5.instances = csr_matrix(data5.instances)
        combined = qp.data.LabelledCollection.join(data4, data5)
        self.assertEqual(len(combined), len(data4) + len(data5))


if __name__ == '__main__':
    unittest.main()
