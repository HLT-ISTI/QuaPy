import unittest

from sklearn.linear_model import LogisticRegression

from quapy.functional import check_prevalence_vector
from quapy.method.aggregative import T50, MAX, X, MS, MS2
from quapy.tests._synthetic import make_dataset


class TestThresholdOptim(unittest.TestCase):

    dataset = make_dataset(n_train=140, n_test=40, n_classes=2, n_features=12, random_state=17, name='synthetic-binary')

    def test_compute_tpr_fpr_edge_cases(self):
        # regression test for the TP/FN vs TP/FP parameter-naming mix-up in _compute_tpr
        model = T50()
        self.assertEqual(model._compute_tpr(TP=5, FN=5), 0.5)
        self.assertEqual(model._compute_tpr(TP=0, FN=0), 1)  # guarded division by zero
        self.assertEqual(model._compute_fpr(FP=3, TN=7), 0.3)
        self.assertEqual(model._compute_fpr(FP=0, TN=0), 0)  # guarded division by zero

    def test_threshold_methods_fit_predict(self):
        learner = LogisticRegression(max_iter=2000)
        learner.fit(*self.dataset.training.Xy)
        for model_cls in [T50, MAX, X, MS, MS2]:
            model = model_cls(learner, fit_classifier=False, val_split=None)
            model.fit(*self.dataset.training.Xy)
            estim_prevalences = model.predict(self.dataset.test.X)
            self.assertTrue(check_prevalence_vector(estim_prevalences), f'{model_cls.__name__} failed')


if __name__ == '__main__':
    unittest.main()
