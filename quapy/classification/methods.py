from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression


class PCALR:
    """
    An example of a classification method that also generates embedded inputs, as those required for QuaNet.
    This example simply combines a Principal Component Analysis (PCA) with Logistic Regression (LR).
    """

    def __init__(self, n_components=300, **kwargs):
        self.n_components = n_components
        self.learner = LogisticRegression(**kwargs)

    def get_params(self):
        params = {'n_components': self.n_components}
        params.update(self.learner.get_params())
        return params

    def set_params(self, **params):
        if 'n_components' in params:
            self.n_components = params['n_components']
            del params['n_components']
        self.learner.set_params(**params)

    def fit(self, X, y):
        self.pca = TruncatedSVD(self.n_components)
        embedded = self.pca.fit_transform(X, y)
        self.learner.fit(embedded, y)
        self.classes_ = self.learner.classes_
        return self

    def predict(self, X):
        embedded = self.transform(X)
        return self.learner.predict(embedded)

    def predict_proba(self, X):
        embedded = self.transform(X)
        return self.learner.predict_proba(embedded)

    def transform(self, X):
        return self.pca.transform(X)
