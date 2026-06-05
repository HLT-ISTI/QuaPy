import time
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from quapy.method.aggregative import PACC
from quapy.model_selection import GridSearchQ
from quapy.protocol import APP
from quapy.tests._synthetic import make_dataset


class ModselTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = make_dataset(
            n_train=220,
            n_test=120,
            n_classes=2,
            n_features=16,
            class_sep=1.8,
            random_state=1,
            name='modsel',
        )
        cls.training, cls.validation = data.training.split_stratified(0.7, random_state=1)

    def test_modsel(self):
        """
        Checks whether a model selection exploration picks the better hyperparameter.
        """
        q = PACC(LogisticRegression(random_state=1, max_iter=5000))
        param_grid = {'classifier__C': [0.000001, 10.0]}
        app = APP(self.validation, sample_size=30, n_prevalences=5, repeats=1, random_state=1)
        q = GridSearchQ(
            q, param_grid, protocol=app, error='mae', refit=False, timeout=-1, verbose=False, n_jobs=-1
        ).fit(*self.training.Xy)

        self.assertEqual(q.best_params_['classifier__C'], 10.0)
        self.assertEqual(q.best_model().get_params()['classifier__C'], 10.0)

    def test_modsel_parallel(self):
        """
        Checks whether sequential and parallel model selection agree on the best parameters.
        """
        q = PACC(LogisticRegression(random_state=1, max_iter=3000))
        param_grid = {'classifier__C': np.logspace(-3, 3, 7), 'classifier__class_weight': ['balanced', None]}
        app = APP(self.validation, sample_size=30, n_prevalences=5, repeats=1, random_state=1)

        def do_gridsearch(n_jobs):
            t_init = time.time()
            modsel = GridSearchQ(
                q, param_grid, protocol=app, error='mae', refit=False, timeout=-1, n_jobs=n_jobs, verbose=False
            ).fit(*self.training.Xy)
            t_end = time.time() - t_init
            return t_end, modsel.best_params_

        _, best_seq = do_gridsearch(n_jobs=1)
        _, best_par = do_gridsearch(n_jobs=-1)

        self.assertEqual(best_seq, best_par)

    def test_modsel_timeout(self):

        class SlowLR(LogisticRegression):
            def fit(self, X, y, sample_weight=None):
                time.sleep(2)
                return super().fit(X, y, sample_weight)

        q = PACC(SlowLR(max_iter=1000))
        param_grid = {'classifier__C': np.logspace(-1, 1, 3)}
        app = APP(self.validation, sample_size=30, n_prevalences=5, repeats=1, random_state=1)

        modsel = GridSearchQ(
            q, param_grid, protocol=app, timeout=1, n_jobs=-1, verbose=False, raise_errors=True
        )
        with self.assertRaises(TimeoutError):
            modsel.fit(*self.training.Xy)

        modsel = GridSearchQ(
            q, param_grid, protocol=app, timeout=1, n_jobs=-1, verbose=False, raise_errors=False
        )
        with self.assertRaises(ValueError):
            modsel.fit(*self.training.Xy)


if __name__ == '__main__':
    unittest.main()
