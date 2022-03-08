from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.sparse import issparse
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR
from statsmodels.miscmodels.ordinal_model import OrderedModel


class OrderedLogisticRegression:
    def __init__(self, model='logit'):
        assert model in ['logit', 'probit'], 'unknown ordered model, valid ones are logit or probit'
        self.model = model

    def fit(self, X, y):
        if issparse(X):
            self.svd = TruncatedSVD(500)
            X = self.svd.fit_transform(X)
        self.learner = OrderedModel(y, X, distr=self.model)
        self.res_prob = self.learner.fit(method='bfgs', disp=False, skip_hessian=True)

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)

    def predict_proba(self, X):
        if issparse(X):
            assert hasattr(self, 'svd'), \
                'X matrix in predict is sparse, but the method has not been fit with sparse type'
            X = self.svd.transform(X)
        return self.res_prob.model.predict(self.res_prob.params, exog=X)


class StackedClassifier:  # aka Funnelling Monolingual
    def __init__(self, base_estimator=LogisticRegression()):
        if not hasattr(base_estimator, 'predict_proba'):
            print('the estimator does not seem to be probabilistic: calibrating')
            base_estimator = CalibratedClassifierCV(base_estimator)
        # self.base = deepcopy(OneVsRestClassifier(base_estimator))
        # self.meta = deepcopy(OneVsRestClassifier(base_estimator))
        self.base = deepcopy(base_estimator)
        self.meta = deepcopy(base_estimator)
        self.norm = StandardScaler()

    def fit(self, X, y):
        self.base.fit(X, y)
        P = self.base.predict_proba(X)
        P = self.norm.fit_transform(P)
        self.meta.fit(P, y)
        return self

    def predict(self, X):
        P = self.base.predict_proba(X)
        P = self.norm.transform(P)
        return self.meta.predict(P)

    def predict_proba(self, X):
        P = self.base.predict_proba(X)
        P = self.norm.transform(P)
        return self.meta.predict_proba(P)


class RegressionQuantification:
    def __init__(self,
                 base_quantifier,
                 regression='svr',
                 val_samples_generator=None,
                 norm=True):

        self.base_quantifier = base_quantifier
        if isinstance(regression, str):
            assert regression in ['ridge', 'svr'], 'unknown regression model'
            if regression == 'ridge':
                self.reg = Ridge(normalize=norm)
            elif regression == 'svr':
                self.reg = MultiOutputRegressor(LinearSVR())
        else:
            self.reg = regression
        # self.reg = MultiTaskLassoCV(normalize=norm)
        # self.reg = KernelRidge(kernel='rbf')
        # self.reg = LassoLarsCV(normalize=norm)
        # self.reg = MultiTaskElasticNetCV(normalize=norm) <- bien
        #self.reg = LinearRegression(normalize=norm) # <- bien
        # self.reg = MultiOutputRegressor(ARDRegression(normalize=norm))  # <- bastante bien, incluso sin norm
        # self.reg = MultiOutputRegressor(BayesianRidge(normalize=False))  # <- bastante bien, incluso sin norm
        # self.reg = MultiOutputRegressor(SGDRegressor())  # lento, no va
        self.regression = regression
        self.val_samples_generator = val_samples_generator
        # self.norm = StandardScaler()
        # self.covs = covs

    def generate_validation_samples(self):
        Xs, ys = [], []
        for instances, prevalence in self.val_samples_generator():
            ys.append(prevalence)
            Xs.append(self.base_quantifier.quantify(instances))
        Xs = np.asarray(Xs)
        ys = np.asarray(ys)
        return Xs, ys

    def fit(self, data):
        print('fitting quantifier')
        if data is not None:
            self.base_quantifier.fit(data)
        print('generating val samples')
        Xs, ys = self.generate_validation_samples()
        # Xs = self.norm.fit_transform(Xs)
        print('fitting regressor')
        self.reg.fit(Xs, ys)
        print('[done]')
        return self

    def quantify(self, instances):
        Xs = self.base_quantifier.quantify(instances).reshape(1, -1)
        # Xs = self.norm.transform(Xs)
        Xs = self.reg.predict(Xs)
        # Xs = self.norm.inverse_transform(Xs)
        adjusted = Xs / Xs.sum()
        # adjusted = np.clip(Xs, 0, 1)
        adjusted = adjusted.flatten()
        return adjusted

    def get_params(self, deep=True):
        return self.base_quantifier.get_params()

    def set_params(self, **params):
        self.base_quantifier.set_params(**params)


class RegressorClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.regressor = LinearSVR()
        # self.regressor = SVR()
        # self.regressor = Ridge(normalize=True)


    def fit(self, X, y):
        self.nclasses = len(np.unique(y))
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        r = self.regressor.predict(X)
        c = np.round(r)
        c[c<0]=0
        c[c>(self.nclasses-1)]=self.nclasses-1
        return c.astype(np.int)

    def predict_proba(self, X):
        r = self.regressor.predict(X)
        nC = len(self.classes_)
        r = np.clip(r, 0, nC - 1)
        dists = np.abs(np.tile(np.arange(nC), (len(r), 1)) - r.reshape(-1,1))
        invdist = 1 - dists
        invdist[invdist < 0] = 0
        return invdist

    @property
    def classes_(self):
        return np.arange(self.nclasses)

    def get_params(self, deep=True):
        return self.regressor.get_params()

    def set_params(self, **params):
        self.regressor.set_params(**params)

