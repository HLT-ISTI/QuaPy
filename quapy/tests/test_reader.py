import os
import tempfile
import unittest

import numpy as np

from quapy.data.reader import from_text, from_sparse, from_csv, reindex_labels, binarize


class TestReader(unittest.TestCase):

    def _write_tmp(self, content, suffix='.txt'):
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        self.addCleanup(os.remove, path)
        return path

    def test_from_text(self):
        path = self._write_tmp('1\tthis is positive\n0\tthis is negative\n')
        sentences, labels = from_text(path, verbose=0)
        self.assertEqual(sentences, ['this is positive', 'this is negative'])
        self.assertEqual(labels, [1, 0])

    def test_from_text_skips_malformed_lines(self):
        # a line without a tab separator should be skipped (and warned about), not raise
        path = self._write_tmp('1\tgood line\nthis line has no label\n0\tanother good line\n')
        sentences, labels = from_text(path, verbose=0)
        self.assertEqual(sentences, ['good line', 'another good line'])
        self.assertEqual(labels, [1, 0])

    def test_from_sparse(self):
        # format: <label> <col:val> <col:val> ...  (1-indexed columns)
        path = self._write_tmp('1 1:0.5 2:1.0\n-1 2:2.0\n', suffix='.dat')
        X, y = from_sparse(path)
        self.assertEqual(X.shape[0], 2)
        np.testing.assert_array_equal(y, np.array([2, 0]))  # labels shifted by +1

    def test_from_csv(self):
        path = self._write_tmp('a,1.0,2.0\nb,3.0,4.0\n', suffix='.csv')
        X, y = from_csv(path)
        np.testing.assert_array_equal(X, np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_array_equal(y, np.array(['a', 'b']))

    def test_reindex_labels(self):
        indexed, classnames = reindex_labels(['B', 'B', 'A', 'C'])
        np.testing.assert_array_equal(indexed, np.array([1, 1, 0, 2]))
        np.testing.assert_array_equal(classnames, np.array(['A', 'B', 'C']))

    def test_binarize(self):
        binarized = binarize([1, 2, 3, 1, 1, 0], pos_class=2)
        np.testing.assert_array_equal(binarized, np.array([0, 1, 0, 0, 0, 0]))


if __name__ == '__main__':
    unittest.main()
