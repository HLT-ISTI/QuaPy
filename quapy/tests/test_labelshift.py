import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from quapy.classification.labelshift import LabelShiftedClassifier
from quapy.method.aggregative import ACC, BBSEhard, BBSEsoft, ImportanceWeightQuantifier, RLLS
from quapy.functional import check_prevalence_vector
from quapy.tests._synthetic import make_dataset


class TestLabelShift(unittest.TestCase):

    dataset = make_dataset(n_train=200, n_test=100, n_classes=3, n_features=10, random_state=7, name='labelshift')

    def test_importance_weight_quantifiers(self):
        Xtr, ytr = self.dataset.training.Xy
        Xte = self.dataset.test.X
        for cls in [BBSEhard, BBSEsoft, RLLS]:
            if cls is RLLS:
                try:
                    import cvxpy  # noqa: F401
                except ImportError:
                    continue

            q = cls(LogisticRegression(max_iter=2000), val_split=3)
            q.fit(Xtr, ytr)

            weights = q.get_importance_weights(Xte)
            prevalence, weights2 = q.quantify_and_weigh(Xte)
            prevalence_direct = q.predict(Xte)

            self.assertTrue(check_prevalence_vector(prevalence))
            np.testing.assert_allclose(weights, weights2)
            np.testing.assert_allclose(prevalence, prevalence_direct)
            # no leftover mutable per-call state (safety against races under concurrent calls)
            self.assertFalse(hasattr(q, 'weights'))
            self.assertFalse(hasattr(q, 'last_w_'))

    def test_label_shifted_classifier_generic_quantifier(self):
        Xtr, ytr = self.dataset.training.Xy
        Xte = self.dataset.test.X

        base_classifier = LogisticRegression(max_iter=2000)
        q = ACC(LogisticRegression(max_iter=2000), val_split=3)
        self.assertNotIsInstance(q, ImportanceWeightQuantifier)

        lsc = LabelShiftedClassifier(base_classifier, q)
        lsc.fit(Xtr, ytr)

        weights = lsc.get_importance_weights(Xte)
        self.assertEqual(weights.shape, (self.dataset.training.n_classes,))
        self.assertTrue(np.all(weights >= lsc.weight_epsilon))

        preds = lsc.predict(Xte)
        probs = lsc.predict_proba(Xte)
        self.assertEqual(len(preds), len(Xte))
        self.assertEqual(probs.shape, (len(Xte), self.dataset.training.n_classes))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

        # the classifier instance passed at construction time must be left untouched
        with self.assertRaises(NotFittedError):
            check_is_fitted(base_classifier)

    def test_label_shifted_classifier_importance_weight_quantifier(self):
        Xtr, ytr = self.dataset.training.Xy
        Xte = self.dataset.test.X

        q = BBSEhard(LogisticRegression(max_iter=2000), val_split=3)
        self.assertIsInstance(q, ImportanceWeightQuantifier)

        lsc = LabelShiftedClassifier(LogisticRegression(max_iter=2000), q)
        lsc.fit(Xtr, ytr)

        weights = lsc.get_importance_weights(Xte)
        adapted = lsc.get_classifier(Xte)
        check_is_fitted(adapted)  # should not raise
        self.assertEqual(set(adapted.get_params()['class_weight'].keys()), set(lsc.classes_))


if __name__ == '__main__':
    unittest.main()
