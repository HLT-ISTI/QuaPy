import unittest
import numpy as np
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


if __name__ == '__main__':
    unittest.main()
