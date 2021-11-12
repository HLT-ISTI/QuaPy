from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression


class LowRankLogisticRegression(BaseEstimator):
    """
    An example of a classification method (i.e., an object that implements `fit`, `predict`, and `predict_proba`)
    that also generates embedded inputs (i.e., that implements `transform`), as those required for
    :class:`quapy.method.neural.QuaNet`. This is a mock method to allow for easily instantiating
    :class:`quapy.method.neural.QuaNet` on array-like real-valued instances.
    The transformation consists of applying :class:`sklearn.decomposition.TruncatedSVD`
    while classification is performed using :class:`sklearn.linear_model.LogisticRegression` on the low-rank space.

    :param n_components: the number of principal components to retain
    :param kwargs: parameters for the
        `Logistic Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`__ classifier
    """

    def __init__(self, n_components=100, **kwargs):
        self.n_components = n_components
        self.learner = LogisticRegression(**kwargs)

    def get_params(self):
        """
        Get hyper-parameters for this estimator.

        :return: a dictionary with parameter names mapped to their values
        """
        params = {'n_components': self.n_components}
        params.update(self.learner.get_params())
        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        :param parameters: a `**kwargs` dictionary with the estimator parameters for
            `Logistic Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`__
            and eventually also `n_components` for `TruncatedSVD`
        """
        params_ = dict(params)
        if 'n_components' in params_:
            self.n_components = params_['n_components']
            del params_['n_components']
        self.learner.set_params(**params_)

    def fit(self, X, y):
        """
        Fit the model according to the given training data. The fit consists of
        fitting `TruncatedSVD` and then `LogisticRegression` on the low-rank representation.

        :param X: array-like of shape `(n_samples, n_features)` with the instances
        :param y: array-like of shape `(n_samples, n_classes)` with the class labels
        :return: `self`
        """
        nF = X.shape[1]
        self.pca = None
        if nF > self.n_components:
            self.pca = TruncatedSVD(self.n_components).fit(X)
        X = self.transform(X)
        self.learner.fit(X, y)
        self.classes_ = self.learner.classes_
        return self

    def predict(self, X):
        """
        Predicts labels for the instances `X` embedded into the low-rank space.

        :param X: array-like of shape `(n_samples, n_features)` instances to classify
        :return: a `numpy` array of length `n` containing the label predictions, where `n` is the number of
            instances in `X`
        """
        X = self.transform(X)
        return self.learner.predict(X)

    def predict_proba(self, X):
        """
        Predicts posterior probabilities for the instances `X` embedded into the low-rank space.

        :param X: array-like of shape `(n_samples, n_features)` instances to classify
        :return: array-like of shape `(n_samples, n_classes)` with the posterior probabilities
        """
        X = self.transform(X)
        return self.learner.predict_proba(X)

    def transform(self, X):
        """
        Returns the low-rank approximation of `X` with `n_components` dimensions, or `X` unaltered if
        `n_components` >= `X.shape[1]`.
        
        :param X: array-like of shape `(n_samples, n_features)` instances to embed
        :return: array-like of shape `(n_samples, n_components)` with the embedded instances
        """
        if self.pca is None:
            return X
        return self.pca.transform(X)
