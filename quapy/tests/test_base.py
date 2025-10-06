import unittest


class ImportTest(unittest.TestCase):
    def test_import(self):
        import quapy as qp
        self.assertIsNotNone(qp.__version__)


if __name__ == '__main__':
    unittest.main()
