from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression


class PCALR:

    def __init__(self, n_components=300, C=10, class_weight=None):
        self.n_components = n_components
        self.learner = LogisticRegression(C=C, class_weight=class_weight, max_iter=1000)

    def get_params(self):
        params = {'n_components': self.n_components}
        params.update(self.learner.get_params())
        return params

    def set_params(self, **params):
        if 'n_components' in params:
            self.n_components = params['n_components']
            del params['n_components']
        self.learner.set_params(**params)

    def fit(self, documents, labels):
        self.pca = TruncatedSVD(self.n_components)
        embedded = self.pca.fit_transform(documents, labels)
        self.learner.fit(embedded, labels)
        self.classes_ = self.learner.classes_
        return self

    def predict(self, documents):
        embedded = self.transform(documents)
        return self.learner.predict(embedded)

    def predict_proba(self, documents):
        embedded = self.transform(documents)
        return self.learner.predict_proba(embedded)

    def transform(self, documents):
        return self.pca.transform(documents)
