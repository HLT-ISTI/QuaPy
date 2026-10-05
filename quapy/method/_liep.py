import warnings

import numpy as np
from sklearn.base import BaseEstimator

import quapy.functional as F
from quapy.method._helper import _labels_to_indices
from quapy.method.aggregative import AggregativeSoftQuantifier


def _normalize(v, eps=1e-12):
    v = np.maximum(np.asarray(v, dtype=float), eps)
    return v / v.sum()


def _bayes_prior_update(probs, new_prior, source_prior, eps=1e-12):
    """
    Applies the standard label-shift prior correction:

        p_new(y|x) \\propto p_old(y|x) * new_prior(y) / source_prior(y)

    :param probs: array of shape `(n_instances, n_classes)` with the (uncorrected) posterior probabilities
    :param new_prior: array of shape `(n_classes,)`, the current estimate of the target prior
    :param source_prior: array of shape `(n_classes,)`, the source (training) prior
    :param eps: numerical stabilizer
    :return: array of shape `(n_instances, n_classes)` with the corrected posterior probabilities
    """
    weights = _normalize(new_prior, eps) / np.maximum(_normalize(source_prior, eps), eps)
    updated = probs * weights[np.newaxis, :]
    updated /= np.maximum(updated.sum(axis=1, keepdims=True), eps)
    return updated


def _confusion_statistic(y_true, y_pred, n_classes, mode='recall', eps=1e-12):
    """
    Computes the minimum diagonal statistic of the confusion matrix, used to choose the fraction of target
    instances to be retained in the high-confidence set (LEIP, Section 4.1).

    :param y_true: array of true label indices in `[0, n_classes)`
    :param y_pred: array of predicted label indices in `[0, n_classes)`
    :param n_classes: number of classes
    :param mode: "recall" (per-true-class recall, i.e., diag(C) / row sums) or "precision" (per-predicted-class
        precision, i.e., diag(C) / column sums)
    :param eps: numerical stabilizer
    :return: float, the minimum, across classes with at least one valid instance, of the chosen statistic
    """
    C = np.zeros((n_classes, n_classes), dtype=float)
    for yt, yp in zip(y_true, y_pred):
        C[yt, yp] += 1.

    denom = C.sum(axis=1) if mode == 'recall' else C.sum(axis=0)
    diag = np.diag(C)
    valid = denom > 0
    if not np.any(valid):
        raise ValueError('no valid classes found in the confusion matrix')

    return float(np.min(diag[valid] / np.maximum(denom[valid], eps)))


class LEIP(AggregativeSoftQuantifier):
    """
    `Label Shift Estimation With Incremental Prior update
    <https://doi.org/10.1137/1.9781611978520.12>`_ (LEIP).

    Zhang, Y., Batista, G., & Kanhere, S.S. (2025). Label Shift Estimation With Incremental Prior Update.
    In Proceedings of the 2025 SIAM International Conference on Data Mining (SDM), pp. 134-142.

    LEIP is a variant of :class:`EMQ` (aka SLD) that estimates the target prior without resorting to an
    iterative Expectation-Maximization procedure. The target instances are first split into a high-confidence
    set (those instances whose maximum posterior probability is at or above a threshold `tau`) and a
    low-confidence set. The high-confidence set is used to obtain an initial classify-and-count estimate of
    the target prior. The low-confidence instances are then relabelled one at a time, in order of decreasing
    confidence, applying at each step the standard label-shift Bayes correction (see the module-level function
    :func:`_bayes_prior_update`) using the running target-prior estimate; each newly assigned pseudo-label
    updates this running estimate before the next instance is processed. Finally, a single Bayes correction is
    applied to the whole target set using the resulting prior, and a hard-label classify-and-count pass on the
    corrected posteriors yields the returned prevalence estimate.

    :param classifier: a scikit-learn's BaseEstimator, or None, in which case the classifier is taken to be
        the one indicated in `qp.environ['DEFAULT_CLS']`

    :param fit_classifier: whether to train the classifier (default is True). Set to False if the
        given classifier has already been trained.

    :param val_split: specifies the data used, when `tau=None`, for estimating the fraction of target
        instances to retain in the high-confidence set (see `threshold_mode`). This specification can be made
        as a float in (0, 1) indicating the proportion of stratified held-out validation set to be extracted
        from the training set; or as an integer (default 5), indicating that the predictions are to be
        generated in a `k`-fold cross-validation manner (with this integer indicating the value for `k`); or
        as a tuple `(X, y)` defining the specific set of data to use for validation. This parameter is ignored
        when a fixed `tau` is provided.

    :param tau: float or None (default). A fixed confidence threshold used to split the target instances into
        the high- and low-confidence sets. If None, the threshold is instead derived, independently for each
        target sample, as the quantile of that sample's own posterior-confidence distribution corresponding to
        a retention fraction estimated from `val_split` (see `threshold_mode`).

    :param threshold_mode: either "recall" (default) or "precision", the statistic computed on the validation
        confusion matrix that determines the fraction of target instances to be retained in the high-confidence
        set (see the module-level function :func:`_confusion_statistic`). Only used when `tau=None`.

    :param count_smoothing: float, additive smoothing applied to the pseudo-label counts used to compute the
        running target-prior estimate (default 0.0, matching the original paper; a small positive value, e.g.,
        1e-8, can instead be used for extra numerical robustness).
    """

    THRESHOLD_MODES = ['recall', 'precision']

    def __init__(self, classifier: BaseEstimator = None, fit_classifier=True, val_split=5, tau=None,
                 threshold_mode='recall', count_smoothing=0.0):

        assert threshold_mode in LEIP.THRESHOLD_MODES, \
            f'invalid {threshold_mode=}; valid ones are {LEIP.THRESHOLD_MODES}'
        assert tau is None or isinstance(tau, (int, float)), f'invalid {tau=}; must be None or a number'
        assert isinstance(count_smoothing, (int, float)) and count_smoothing >= 0, \
            f'invalid {count_smoothing=}; must be a non-negative number'

        super().__init__(classifier, fit_classifier, val_split)
        self.tau = tau
        self.threshold_mode = threshold_mode
        self.count_smoothing = count_smoothing

    def _check_init_parameters(self):
        if self.tau is None:
            if self.val_split is None:
                raise ValueError(
                    'LEIP requires validation predictions to estimate the retention threshold when tau=None; '
                    'please set val_split to an integer, float, or validation tuple, or provide a fixed tau.'
                )
        elif self.val_split is not None:
            warnings.warn(f'a fixed {self.tau=} was set; {self.val_split=} will be ignored, since LEIP only '
                           f'needs validation data to estimate a retention threshold when tau is not fixed.')

    def classify(self, X):
        """
        Provides the posterior probabilities for the given instances.

        :param X: array-like of shape `(n_instances, n_dimensions,)`
        :return: np.ndarray of shape `(n_instances, n_classes,)` with posterior probabilities
        """
        return self.classifier.predict_proba(X)

    def classifier_fit_predict(self, X, y):
        classif_predictions = super().classifier_fit_predict(X, y)
        self.train_prevalence = F.prevalence_from_labels(y, classes=self.classes_)
        return classif_predictions

    def aggregation_fit(self, classif_predictions, labels):
        """
        Trains the aggregation function of LEIP. When a fixed `tau` was not provided at construction time,
        this estimates, from the validation predictions, the fraction of target instances that should be
        retained in the high-confidence set (see :func:`_confusion_statistic`).

        :param classif_predictions: array-like with the posterior probabilities of the validation instances
        :param labels: array-like with the true labels associated to each classifier prediction
        """
        if self.tau is None:
            n_classes = len(self.classes_)
            y_val_idx = _labels_to_indices(labels, self.classes_)
            y_val_pred_idx = classif_predictions.argmax(axis=1)
            self.retain_fraction_ = _confusion_statistic(
                y_val_idx, y_val_pred_idx, n_classes, mode=self.threshold_mode
            )
        else:
            self.retain_fraction_ = None

    def aggregate(self, classif_posteriors):
        prevalence, _ = self._leip(classif_posteriors)
        return prevalence

    def predict_proba(self, instances):
        """
        Returns the posterior probabilities updated by the final Bayes correction applied by LEIP.

        :param instances: np.ndarray of shape `(n_instances, n_dimensions)`
        :return: np.ndarray of shape `(n_instances, n_classes)`
        """
        classif_posteriors = self.classify(instances)
        _, corrected_posteriors = self._leip(classif_posteriors)
        return corrected_posteriors

    def _leip(self, target_probs):
        n_classes = len(self.classes_)
        source_prior = self.train_prevalence

        target_conf = target_probs.max(axis=1)
        target_top = target_probs.argmax(axis=1)

        if self.tau is not None:
            tau = self.tau
        else:
            retain_fraction = float(np.clip(self.retain_fraction_, 0., 1.))
            if retain_fraction <= 0:
                tau = np.inf
            elif retain_fraction >= 1:
                tau = -np.inf
            else:
                tau = np.quantile(target_conf, 1. - retain_fraction)
        self.tau_ = tau

        # high-confidence set A: an initial classify-and-count estimate of the target prior
        A_mask = target_conf >= tau
        counts = np.full(n_classes, self.count_smoothing, dtype=float)
        if np.any(A_mask):
            counts += np.bincount(target_top[A_mask], minlength=n_classes)
            current_prior = counts / counts.sum()
        else:
            current_prior = source_prior.copy()

        # low-confidence set B, relabelled incrementally in order of decreasing confidence
        B_indices = np.where(~A_mask)[0]
        B_indices = B_indices[np.argsort(-target_conf[B_indices])]

        for idx in B_indices:
            corrected = _bayes_prior_update(target_probs[idx:idx + 1], current_prior, source_prior)[0]
            pseudo_label = int(np.argmax(corrected))
            counts[pseudo_label] += 1.
            current_prior = counts / counts.sum()

        self.intermediate_prior_ = current_prior.copy()

        # final full-batch Bayes correction and hard-label classify-and-count pass
        corrected_posteriors = _bayes_prior_update(target_probs, current_prior, source_prior)
        final_labels = corrected_posteriors.argmax(axis=1)
        prevalence = F.prevalence_from_labels(final_labels, classes=np.arange(n_classes))

        return prevalence, corrected_posteriors
