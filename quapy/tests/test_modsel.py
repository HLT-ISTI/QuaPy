import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

import quapy as qp
from quapy.method.aggregative import PACC
from quapy.model_selection import GridSearchQ
from quapy.protocol import APP
import time


class ModselTestCase(unittest.TestCase):

    def test_modsel(self):
        """
        Checks whether a model selection exploration takes a good hyperparameter
        """

        q = PACC(LogisticRegression(random_state=1, max_iter=5000))

        data = qp.datasets.fetch_reviews('imdb', tfidf=True, min_df=10).reduce(random_state=1)
        training, validation = data.training.split_stratified(0.7, random_state=1)

        param_grid = {'classifier__C': [0.000001, 10.]}
        app = APP(validation, sample_size=100, random_state=1)
        q = GridSearchQ(
            q, param_grid, protocol=app, error='mae', refit=True, timeout=-1, verbose=True
        ).fit(training)
        print('best params', q.best_params_)
        print('best score', q.best_score_)

        self.assertEqual(q.best_params_['classifier__C'], 10.0)
        self.assertEqual(q.best_model().get_params()['classifier__C'], 10.0)

    def test_modsel_parallel(self):
        """
        Checks whether a parallelized model selection actually is faster than a sequential exploration but
        obtains the same optimal parameters
        """

        q = PACC(LogisticRegression(random_state=1, max_iter=5000))

        data = qp.datasets.fetch_reviews('imdb', tfidf=True, min_df=10).reduce(n_train=500, random_state=1)
        training, validation = data.training.split_stratified(0.7, random_state=1)

        param_grid = {'classifier__C': np.logspace(-3,3,7)}
        app = APP(validation, sample_size=100, random_state=1)

        print('starting model selection in sequential exploration')
        tinit = time.time()
        modsel = GridSearchQ(
            q, param_grid, protocol=app, error='mae', refit=True, timeout=-1, n_jobs=1, verbose=True
        ).fit(training)
        tend_seq = time.time()-tinit
        best_c_seq = modsel.best_params_['classifier__C']
        print(f'[done] took {tend_seq:.2f}s best C = {best_c_seq}')

        print('starting model selection in parallel exploration')
        tinit = time.time()
        modsel = GridSearchQ(
            q, param_grid, protocol=app, error='mae', refit=True, timeout=-1, n_jobs=-1, verbose=True
        ).fit(training)
        tend_par = time.time() - tinit
        best_c_par = modsel.best_params_['classifier__C']
        print(f'[done] took {tend_par:.2f}s best C = {best_c_par}')

        self.assertEqual(best_c_seq, best_c_par)
        self.assertLess(tend_par, tend_seq)


    def test_modsel_timeout(self):

        class SlowLR(LogisticRegression):
            def fit(self, X, y, sample_weight=None):
                import time
                time.sleep(10)
                super(SlowLR, self).fit(X, y, sample_weight)

        q = PACC(SlowLR())

        data = qp.datasets.fetch_reviews('imdb', tfidf=True, min_df=10).reduce(random_state=1)
        training, validation = data.training.split_stratified(0.7, random_state=1)

        param_grid = {'classifier__C': np.logspace(-1,1,3)}
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
