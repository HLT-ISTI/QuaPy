import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from quapy.classification.calibration import NBVSCalibration, BCTSCalibration, TSCalibration, VSCalibration
from quapy.tests._synthetic import make_labelled_collection


class TestCalibration(unittest.TestCase):

    data = make_labelled_collection(n_samples=200, n_features=10, n_classes=3, random_state=23)

    def test_calibration_methods_fit_predict_proba(self):
        X, y = self.data.Xy
        for calib_cls in [NBVSCalibration, BCTSCalibration, TSCalibration, VSCalibration]:
            model = calib_cls(LogisticRegression(max_iter=2000), val_split=5)
            model.fit(X, y)
            posteriors = model.predict_proba(X)
            self.assertEqual(posteriors.shape, (len(y), self.data.n_classes))
            np.testing.assert_allclose(posteriors.sum(axis=1), 1.0, rtol=1e-5,
                                        err_msg=f'{calib_cls.__name__} posteriors do not sum to 1')
            predictions = model.predict(X)
            self.assertEqual(len(predictions), len(y))

    def test_calibration_with_float_val_split(self):
        X, y = self.data.Xy
        model = BCTSCalibration(LogisticRegression(max_iter=2000), val_split=0.3, random_state=0)
        model.fit(X, y)
        posteriors = model.predict_proba(X)
        self.assertEqual(posteriors.shape, (len(y), self.data.n_classes))


if __name__ == '__main__':
    unittest.main()
