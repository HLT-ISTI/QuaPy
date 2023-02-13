import unittest

from sklearn.linear_model import LogisticRegression

import quapy as qp
from quapy.method.aggregative import *



class HierarchyTestCase(unittest.TestCase):

    def test_aggregative(self):
        lr = LogisticRegression()
        for m in [CC(lr), PCC(lr), ACC(lr), PACC(lr)]:
            self.assertEqual(isinstance(m, AggregativeQuantifier), True)

    def test_binary(self):
        lr = LogisticRegression()
        for m in [HDy(lr)]:
            self.assertEqual(isinstance(m, BinaryQuantifier), True)

    def test_probabilistic(self):
        lr = LogisticRegression()
        for m in [CC(lr), ACC(lr)]:
            self.assertEqual(isinstance(m, AggregativeProbabilisticQuantifier), False)
        for m in [PCC(lr), PACC(lr)]:
            self.assertEqual(isinstance(m, AggregativeProbabilisticQuantifier), True)


if __name__ == '__main__':
    unittest.main()
