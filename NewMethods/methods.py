import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import quapy as qp
from typing import Union

from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier, BinaryQuantifier
from quapy.method.aggregative import PACC, EMQ, HDy
import quapy.functional as F
from tqdm import tqdm
from scipy.sparse import issparse, csr_matrix
import scipy


class PACCSLD(PACC):
    """
    This method combines the EMQ improved posterior probabilities with PACC.
    Note: the posterior probabilities are re-calibrated with EMQ only during prediction, and not also during fit since,
    for PACC, the validation split is known to have the same prevalence as the training set (this is because the split
    is stratified) and thus the posterior probabilities should not be re-calibrated for a different prior (it actually
    happens to degrades performance).
    """

    def fit(self, data: qp.data.LabelledCollection, fit_learner=True, val_split:Union[float, int, qp.data.LabelledCollection]=0.4):
        self.train_prevalence = F.prevalence_from_labels(data.labels, data.n_classes)
        return super(PACCSLD, self).fit(data, fit_learner, val_split)

    def aggregate(self, classif_posteriors):
        priors, posteriors = EMQ.EM(self.train_prevalence, classif_posteriors, epsilon=1e-4)
        return super(PACCSLD, self).aggregate(posteriors)


class HDySLD(HDy):
    """
        This method combines the EMQ improved posterior probabilities with HDy.
        Note: [same as PACCSLD]
        """
    def fit(self, data: qp.data.LabelledCollection, fit_learner=True,
            val_split: Union[float, int, qp.data.LabelledCollection] = 0.4):
        self.train_prevalence = F.prevalence_from_labels(data.labels, data.n_classes)
        return super(HDySLD, self).fit(data, fit_learner, val_split)

    def aggregate(self, classif_posteriors):
        priors, posteriors = EMQ.EM(self.train_prevalence, classif_posteriors, epsilon=1e-4)
        return super(HDySLD, self).aggregate(posteriors)



class AveragePoolQuantification(BinaryQuantifier):
    def __init__(self, learner, sample_size, trials, n_components=-1, zscore=False):
        self.learner = learner
        self.sample_size = sample_size
        self.trials = trials

        self.do_zscore = zscore
        self.zscore = StandardScaler() if self.do_zscore else None

        self.do_pca = n_components>0
        self.pca = PCA(n_components) if self.do_pca else None

    def fit(self, data: LabelledCollection):
        training, validation = data.split_stratified(train_prop=0.7)

        X, y = [], []

        nprevpoints = F.get_nprevpoints_approximation(self.trials, data.n_classes)
        for sample in tqdm(
                training.artificial_sampling_generator(self.sample_size, n_prevalences=nprevpoints, repeats=1),
                desc='generating averages'
        ):
            X.append(sample.instances.mean(axis=0))
            y.append(sample.prevalence()[1])
        while len(X) < self.trials:
            sample = training.sampling(self.sample_size, F.uniform_simplex_sampling(data.n_classes))
            X.append(sample.instances.mean(axis=0))
            y.append(sample.prevalence())
        X = np.asarray(np.vstack(X))
        y = np.asarray(y)

        if self.do_pca:
            X = self.pca.fit_transform(X)
            print(X.shape)

        if self.do_zscore:
            X = self.zscore.fit_transform(X)

        print('training regressor...')
        self.regressor = self.learner.fit(X, y)

        # correction at 0:
        print('getting corrections...')
        X0 = np.asarray(np.vstack([validation.sampling(self.sample_size, 0., shuffle=False).instances.mean(axis=0) for _ in range(100)]))
        X1 = np.asarray(np.vstack([validation.sampling(self.sample_size, 1., shuffle=False).instances.mean(axis=0) for _ in range(100)]))

        if self.do_pca:
            X0 = self.pca.transform(X0)
            X1 = self.pca.transform(X1)

        if self.do_zscore:
            X0 = self.zscore.transform(X0)
            X1 = self.zscore.transform(X1)

        self.correction_0 = self.regressor.predict(X0).mean()
        self.correction_1 = self.regressor.predict(X1).mean()

        print('correction-0', self.correction_0)
        print('correction-1', self.correction_1)
        print('done')

    def quantify(self, instances):
        ave = np.asarray(instances.mean(axis=0))

        if self.do_pca:
            ave = self.pca.transform(ave)
        if self.do_zscore:
            ave = self.zscore.transform(ave)
        phat = self.regressor.predict(ave).item()
        phat = np.clip((phat-self.correction_0)/(self.correction_1-self.correction_0), 0, 1)
        return np.asarray([1-phat, phat])

    def set_params(self, **parameters):
        self.learner.set_params(**parameters)

    def get_params(self, deep=True):
        return self.learner.get_params(deep=deep)


class WinnowOrthogonal(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes_ = np.asarray(sorted(np.unique(y)))
        w1 = np.asarray(X[y == 0].mean(axis=0)).flatten()
        w2 = np.asarray(X[y == 1].mean(axis=0)).flatten()
        diff = w2 - w1
        orth = np.ones_like(diff)
        orth[0] = -diff[1:].sum() / diff[0]
        orth /= np.linalg.norm(orth)
        self.w = orth
        self.b = w1.dot(orth)
        return self

    def decision_function(self, X):
        if issparse(X):
            Z = X.dot(csr_matrix(self.w).T).toarray().flatten()
            return Z - self.b
        else:
            return np.matmul(X, self.w) - self.b

    def predict(self, X):
        return 1 * (self.decision_function(X) > 0)

    def split(self, X, y):
        s = self.predict(X)
        X0a = X[np.logical_and(y == 0, s == 0)]
        X0b = X[np.logical_and(y == 0, s == 1)]
        X1a = X[np.logical_and(y == 1, s == 0)]
        X1b = X[np.logical_and(y == 1, s == 1)]
        y0a = np.zeros(X0a.shape[0], dtype=np.int)
        y0b = np.zeros(X0b.shape[0], dtype=np.int)
        y1a = np.ones(X1a.shape[0], dtype=np.int)
        y1b = np.ones(X1b.shape[0], dtype=np.int)
        return X0a, X0b, X1a, X1b, y0a, y0b, y1a, y1b

    def get_params(self):
        return {}

    def set_params(self, **params):
        pass
