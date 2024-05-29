import pickle

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from LeQua2024._lequa2024 import fetch_lequa2024
from quapy.data import LabelledCollection
from quapy.protocol import AbstractProtocol
from quapy.method.base import BaseQuantifier
import quapy.functional as F
from tqdm import tqdm
from scripts.evaluate import normalized_match_distance, match_distance


def projection_simplex_sort(unnormalized_arr) -> np.ndarray:
    """Projects a point onto the probability simplex.
    [This code is taken from the devel branch, that will correspond to the future QuaPy 0.1.9]

    The code is adapted from Mathieu Blondel's BSD-licensed
    `implementation <https://gist.github.com/mblondel/6f3b7aaad90606b98f71>`_
    (see function `projection_simplex_sort` in their repo) which is accompanying the paper

    Mathieu Blondel, Akinori Fujino, and Naonori Ueda.
    Large-scale Multiclass Support Vector Machine Training via Euclidean Projection onto the Simplex,
    ICPR 2014, `URL <http://www.mblondel.org/publications/mblondel-icpr2014.pdf>`_

    :param `unnormalized_arr`: point in n-dimensional space, shape `(n,)`
    :return: projection of `unnormalized_arr` onto the (n-1)-dimensional probability simplex, shape `(n,)`
    """
    unnormalized_arr = np.asarray(unnormalized_arr)
    n = len(unnormalized_arr)
    u = np.sort(unnormalized_arr)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    return np.maximum(unnormalized_arr - theta, 0)


class RegressionToSimplex(BaseEstimator):
    """
    A very simple regressor of probability distributions.
    Internally, this class works by invoking an SVR regressor multioutput
    followed by a mapping onto the probability simplex.

    :param C: regularziation parameter for SVR
    """

    def __init__(self, C=1):
        self.C = C

    def fit(self, X, y):
        """
        Learns the correction

        :param X: array-like of shape `(n_instances, n_classes)` with uncorrected prevalence vectors
        :param y: array-like of shape `(n_instances, n_classes)` with true prevalence vectors
        :return: self
        """
        self.reg = MultiOutputRegressor(SVR(C=self.C), n_jobs=-1)
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        """
        Corrects the a vector of prevalence values

        :param X: array-like of shape `(n_classes,)` with one vector of uncorrected prevalence values
        :return: array-like of shape `(n_classes,)` with one vector of corrected prevalence values
        """
        y_ = self.reg.predict(X)
        y_ = np.asarray([projection_simplex_sort(y_i) for y_i in y_])
        return y_


class KDEyRegressor(BaseQuantifier):
    """
    This class implements a regressor-based correction on top of a quantifier.
    The quantifier is taken to be KDEy-ML, which is considered to be already trained (this
    method simply loads a pickled object).
    The method then optimizes a regressor that corrects prevalence vectors using the
    validation samples as training data.
    The regressor is based on a multioutput SVR and relies on a post-processing to guarantee
    that the output lies on the probability simplex (see also RegressionToSimplex)
    """

    def __init__(self, kde_path, Cs=np.logspace(-3,3,7)):
        self.kde_path = kde_path
        self.Cs = Cs

    def fit(self, val_data: AbstractProtocol):
        print(f'loading kde from {self.kde_path}')
        self.kdey = pickle.load(open(self.kde_path, 'rb'))

        print('representing val data with kde')
        pbar = tqdm(val_data(), total=val_data.total())
        Xs, Ys = [], []
        for sample, prev in pbar:
            prev_hat = self.kdey.quantify(sample)
            Xs.append(prev_hat)
            Ys.append(prev)

        Xs = np.asarray(Xs)
        Ys = np.asarray(Ys)

        def scorer(estimator, X, y):
            y_hat = estimator.predict(X)
            md = normalized_match_distance(y, y_hat)
            return (-md)

        grid = {'C': self.Cs}
        optim = GridSearchCV(
            RegressionToSimplex(), param_grid=grid, scoring=scorer, verbose=0, cv=10, n_jobs=64
        ).fit(Xs, Ys)
        self.regressor = optim.best_estimator_
        return self

    def quantify(self, instances):
        prev_hat = self.kdey.quantify(instances)
        return self.regressor.predict([prev_hat])[0]


if __name__ == '__main__':
    train, gen_val, _ = fetch_lequa2024(task='T3', data_home='./data', merge_T3=True)
    kdey_r = KDEyRegressor('./models/T3/KDEy-ML.pkl')
    kdey_r.fit(gen_val)
    pickle.dump(kdey_r, open('./models/T3/KDEyRegressor.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

