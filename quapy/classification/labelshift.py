import numpy as np
from sklearn.base import BaseEstimator, clone

import quapy.functional as F


class LabelShiftedClassifier(BaseEstimator):
    """
    Adapts a classifier to a shifted (unlabelled) target distribution by means of a `quapy` quantifier.

    Given a base classifier `h` and a quantifier `q`, this wrapper: (i) at `fit` time, fits `q` on the labelled
    training data and keeps a copy of this data; (ii) at prediction time, for a given batch of (unlabelled)
    target instances, estimates the importance weights :math:`w_y=Q(y)/P(y)` that would explain the shift
    between the training distribution and the target one (directly, if `q` is an
    :class:`quapy.method.aggregative.ImportanceWeightQuantifier`, or otherwise by dividing `q`'s prevalence
    estimate by the training prevalence); and (iii) retrains a fresh copy of `h`, reweighted according to `w`
    via `class_weight`, on the original training data, before delegating the classification of the target
    instances to this adapted classifier.

    Note that a fresh copy of `h` is retrained for every batch of target instances passed to `predict`,
    `predict_proba`, or `get_classifier`, since the adaptation is specific to that batch's estimated shift; `q`
    and its own internal classifier (if any) are otherwise left untouched. `h` itself is required to accept a
    `class_weight` parameter in its constructor (e.g., `sklearn.linear_model.LogisticRegression`).

    :param classifier: a scikit-learn classifier that accepts a `class_weight` parameter in its constructor
    :param quantifier: a `quapy` quantifier, used to estimate the target prevalence (or, if it is an
        :class:`quapy.method.aggregative.ImportanceWeightQuantifier`, the importance weights directly)
    :param weight_epsilon: float, a small positive floor applied to the estimated importance weights (default
        1e-4), so that a class with training support is never assigned a `class_weight` of exactly 0 (which
        would silently drop it from the retraining loss)
    """

    def __init__(self, classifier, quantifier, weight_epsilon=1e-4):
        self.classifier = classifier
        self.quantifier = quantifier
        self.weight_epsilon = weight_epsilon

    def fit(self, X, y):
        """
        Fits the internal quantifier on the training data, and stores the training data for the subsequent,
        per-target-batch, classifier adaptation.

        :param X: array-like of shape `(n_samples, n_features)` with the training instances
        :param y: array-like of shape `(n_samples,)` with the class labels
        :return: self
        """
        self.quantifier.fit(X, y)
        self.classes_ = self.quantifier.classes_
        self.train_prevalence_ = F.prevalence_from_labels(y, classes=self.classes_)
        self.X_, self.y_ = X, y
        return self

    def get_importance_weights(self, X):
        """
        Estimates the vector of importance weights :math:`w_y=Q(y)/P(y)` that would explain the shift between
        the training distribution and the distribution of the given target instances. This method does not
        mutate any internal state, so it is safe to call concurrently for different batches of target
        instances.

        :param X: array-like of shape `(n_samples, n_features)` with the (unlabelled) target instances
        :return: np.ndarray of shape `(n_classes,)`
        """
        from quapy.method.aggregative import ImportanceWeightQuantifier
        if isinstance(self.quantifier, ImportanceWeightQuantifier):
            weights = self.quantifier.get_importance_weights(X)
        else:
            test_prevalence = self.quantifier.predict(X)
            weights = test_prevalence / np.maximum(self.train_prevalence_, self.weight_epsilon)
        return np.clip(weights, self.weight_epsilon, None)

    def get_classifier(self, X):
        """
        Returns a fresh classifier instance, sharing `classifier`'s hyperparameters, retrained on the original
        training data with `class_weight` set according to the importance weights estimated for the given
        target instances. The classifier passed at construction time, and the one used internally by the
        quantifier, are left untouched.

        :param X: array-like of shape `(n_samples, n_features)` with the (unlabelled) target instances
        :return: a fitted scikit-learn classifier
        """
        weights = self.get_importance_weights(X)
        class_weight = dict(zip(self.classes_, weights))
        adapted = clone(self.classifier)
        adapted.set_params(class_weight=class_weight)
        adapted.fit(self.X_, self.y_)
        return adapted

    def predict(self, X):
        """
        Adapts the classifier to the target instances in `X`, and returns its label predictions for `X`.
        Adaptation implies retraining. If you intend to generate inferences over many samples from the same
        suspected distribution, then call the `get_classifier` method once, and use the returned classifier
        over such samples.

        :param X: array-like of shape `(n_samples, n_features)` with the (unlabelled) target instances
        :return: array-like of shape `(n_samples,)` with the class label predictions
        """
        return self.get_classifier(X).predict(X)

    def predict_proba(self, X):
        """
        Adapts the classifier to the target instances in `X`, and returns its posterior probabilities for `X`.
        Adaptation implies retraining. If you intend to generate inferences over many samples from the same
        suspected distribution, then call the `get_classifier` method once, and use the returned classifier
        over such samples.

        :param X: array-like of shape `(n_samples, n_features)` with the (unlabelled) target instances
        :return: array-like of shape `(n_samples, n_classes)` with posterior probabilities
        """
        return self.get_classifier(X).predict_proba(X)
