import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import quapy as qp
from quapy.method.aggregative import PACC
from quapy.model_selection import GridSearchQ
from quapy.protocol import APP
import time


class ModselTestCase(unittest.TestCase):

    def test_modsel(self):

        q = PACC(LogisticRegression(random_state=1, max_iter=5000))

        data = qp.datasets.fetch_reviews('imdb', tfidf=True, min_df=10)
        training, validation = data.training.split_stratified(0.7, random_state=1)

        param_grid = {'classifier__C': np.logspace(-3,3,7)}
        app = APP(validation, sample_size=100, random_state=1)
        q = GridSearchQ(
            q, param_grid, protocol=app, error='mae', refit=True, timeout=-1, verbose=True
        ).fit(training)
        print('best params', q.best_params_)
        print('best score', q.best_score_)

        self.assertEqual(q.best_params_['classifier__C'], 10.0)
        self.assertEqual(q.best_model().get_params()['classifier__C'], 10.0)

    def test_modsel_parallel(self):

        q = PACC(LogisticRegression(random_state=1, max_iter=5000))

        data = qp.datasets.fetch_reviews('imdb', tfidf=True, min_df=10)
        training, validation = data.training.split_stratified(0.7, random_state=1)
        # test = data.test

        param_grid = {'classifier__C': np.logspace(-3,3,7)}
        app = APP(validation, sample_size=100, random_state=1)
        q = GridSearchQ(
            q, param_grid, protocol=app, error='mae', refit=True, timeout=-1, n_jobs=-1, verbose=True
        ).fit(training)
        print('best params', q.best_params_)
        print('best score', q.best_score_)

        self.assertEqual(q.best_params_['classifier__C'], 10.0)
        self.assertEqual(q.best_model().get_params()['classifier__C'], 10.0)

    def test_modsel_parallel_speedup(self):
        class SlowLR(LogisticRegression):
            def fit(self, X, y, sample_weight=None):
                time.sleep(1)
                return super(SlowLR, self).fit(X, y, sample_weight)

        q = PACC(SlowLR(random_state=1, max_iter=5000))

        data = qp.datasets.fetch_reviews('imdb', tfidf=True, min_df=10)
        training, validation = data.training.split_stratified(0.7, random_state=1)

        param_grid = {'classifier__C': np.logspace(-3, 3, 7)}
        app = APP(validation, sample_size=100, random_state=1)

        tinit = time.time()
        GridSearchQ(
            q, param_grid, protocol=app, error='mae', refit=False, timeout=-1, n_jobs=1, verbose=True
        ).fit(training)
        tend_nooptim = time.time()-tinit

        tinit = time.time()
        GridSearchQ(
            q, param_grid, protocol=app, error='mae', refit=False, timeout=-1, n_jobs=-1, verbose=True
        ).fit(training)
        tend_optim = time.time() - tinit

        print(f'parallel training took {tend_optim:.4f}s')
        print(f'sequential training took {tend_nooptim:.4f}s')

        self.assertEqual(tend_optim < (0.5*tend_nooptim), True)

    def test_modsel_timeout(self):

        class SlowLR(LogisticRegression):
            def fit(self, X, y, sample_weight=None):
                import time
                time.sleep(10)
                super(SlowLR, self).fit(X, y, sample_weight)

        q = PACC(SlowLR())

        data = qp.datasets.fetch_reviews('imdb', tfidf=True, min_df=10)
        training, validation = data.training.split_stratified(0.7, random_state=1)
        # test = data.test

        param_grid = {'classifier__C': np.logspace(-3,3,7)}
        app = APP(validation, sample_size=100, random_state=1)

        print('Expecting TimeoutError to be raised')
        modsel = GridSearchQ(
            q, param_grid, protocol=app, timeout=3, n_jobs=-1, verbose=True, raise_errors=True
        )
        with self.assertRaises(TimeoutError):
            modsel.fit(training)

        print('Expecting ValueError to be raised')
        modsel = GridSearchQ(
            q, param_grid, protocol=app, timeout=3, n_jobs=-1, verbose=True, raise_errors=False
        )
        with self.assertRaises(ValueError):
            # this exception is not raised because of the timeout, but because no combination of hyperparams
            # succedded (in this case, a ValueError is raised, regardless of "raise_errors"
            modsel.fit(training)


if __name__ == '__main__':
    unittest.main()
