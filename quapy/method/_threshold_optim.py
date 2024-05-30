from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
import quapy as qp
import quapy.functional as F
from quapy.data import LabelledCollection
from quapy.method.aggregative import BinaryAggregativeQuantifier


class ThresholdOptimization(BinaryAggregativeQuantifier):
    """
    Abstract class of Threshold Optimization variants for :class:`ACC` as proposed by
    `Forman 2006 <https://dl.acm.org/doi/abs/10.1145/1150402.1150423>`_ and
    `Forman 2008 <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_.
    The goal is to bring improved stability to the denominator of the adjustment.
    The different variants are based on different heuristics for choosing a decision threshold
    that would allow for more true positives and many more false positives, on the grounds this
    would deliver larger denominators.

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`, defaults 5), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, classifier: BaseEstimator=None, val_split=None, n_jobs=None):
        self.classifier = qp._get_classifier(classifier)
        self.val_split = val_split
        self.n_jobs = qp._get_njobs(n_jobs)

    @abstractmethod
    def condition(self, tpr, fpr) -> float:
        """
        Implements the criterion according to which the threshold should be selected.
        This function should return the (float) score to be minimized.

        :param tpr: float, true positive rate
        :param fpr: float, false positive rate
        :return: float, a score for the given `tpr` and `fpr`
        """
        ...

    def discard(self, tpr, fpr) -> bool:
        """
        Indicates whether a combination of tpr and fpr should be discarded

        :param tpr: float, true positive rate
        :param fpr: float, false positive rate
        :return: true if the combination is to be discarded, false otherwise
        """
        return (tpr - fpr) == 0


    def _eval_candidate_thresholds(self, decision_scores, y):
        """
        Seeks for the best `tpr` and `fpr` according to the score obtained at different
        decision thresholds. The scoring function is implemented in function `_condition`.

        :param decision_scores: array-like with the classification scores
        :param y: predicted labels for the validation set (or for the training set via `k`-fold cross validation)
        :return: best `tpr` and `fpr` and `threshold` according to `_condition`
        """
        candidate_thresholds = np.unique(decision_scores)

        candidates = []
        scores = []
        for candidate_threshold in candidate_thresholds:
            y_ = self.classes_[1 * (decision_scores >= candidate_threshold)]
            TP, FP, FN, TN = self._compute_table(y, y_)
            tpr = self._compute_tpr(TP, FN)
            fpr = self._compute_fpr(FP, TN)
            if not self.discard(tpr, fpr):
                candidate_score = self.condition(tpr, fpr)
                candidates.append([tpr, fpr, candidate_threshold])
                scores.append(candidate_score)

        if len(candidates) == 0:
            # if no candidate gives rise to a valid combination of tpr and fpr, this method defaults to the standard
            # classify & count; this is akin to assign tpr=1, fpr=0, threshold=0
            tpr, fpr, threshold = 1, 0, 0
            candidates.append([tpr, fpr, threshold])
            scores.append(0)

        candidates = np.asarray(candidates)
        candidates = candidates[np.argsort(scores)]  # sort candidates by candidate_score

        return candidates

    def aggregate_with_threshold(self, classif_predictions, tprs, fprs, thresholds):
        # This function performs the adjusted count for given tpr, fpr, and threshold.
        # Note that, due to broadcasting, tprs, fprs, and thresholds could be arrays of length > 1
        prevs_estims = np.mean(classif_predictions[:, None] >= thresholds, axis=0)
        prevs_estims = (prevs_estims - fprs) / (tprs - fprs)
        prevs_estims = F.as_binary_prevalence(prevs_estims, clip_if_necessary=True)
        return prevs_estims.squeeze()

    def _compute_table(self, y, y_):
        TP = np.logical_and(y == y_, y == self.pos_label).sum()
        FP = np.logical_and(y != y_, y == self.neg_label).sum()
        FN = np.logical_and(y != y_, y == self.pos_label).sum()
        TN = np.logical_and(y == y_, y == self.neg_label).sum()
        return TP, FP, FN, TN

    def _compute_tpr(self, TP, FP):
        if TP + FP == 0:
            return 1
        return TP / (TP + FP)

    def _compute_fpr(self, FP, TN):
        if FP + TN == 0:
            return 0
        return FP / (FP + TN)

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        decision_scores, y = classif_predictions.Xy
        # the standard behavior is to keep the best threshold only
        self.tpr, self.fpr, self.threshold = self._eval_candidate_thresholds(decision_scores, y)[0]
        return self

    def aggregate(self, classif_predictions: np.ndarray):
        # the standard behavior is to compute the adjusted count using the best threshold found
        return self.aggregate_with_threshold(classif_predictions, self.tpr, self.fpr, self.threshold)


class T50(ThresholdOptimization):
    """
    Threshold Optimization variant for :class:`ACC` as proposed by
    `Forman 2006 <https://dl.acm.org/doi/abs/10.1145/1150402.1150423>`_ and
    `Forman 2008 <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_ that looks
    for the threshold that makes `tpr` closest to 0.5.
    The goal is to bring improved stability to the denominator of the adjustment.

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`, defaults 5), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, classifier: BaseEstimator=None, val_split=5):
        super().__init__(classifier, val_split)

    def condition(self, tpr, fpr) -> float:
        return abs(tpr - 0.5)


class MAX(ThresholdOptimization):
    """
    Threshold Optimization variant for :class:`ACC` as proposed by
    `Forman 2006 <https://dl.acm.org/doi/abs/10.1145/1150402.1150423>`_ and
    `Forman 2008 <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_ that looks
    for the threshold that maximizes `tpr-fpr`.
    The goal is to bring improved stability to the denominator of the adjustment.

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`, defaults 5), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, classifier: BaseEstimator=None, val_split=5):
        super().__init__(classifier, val_split)

    def condition(self, tpr, fpr) -> float:
        # MAX strives to maximize (tpr - fpr), which is equivalent to minimize (fpr - tpr)
        return (fpr - tpr)


class X(ThresholdOptimization):
    """
    Threshold Optimization variant for :class:`ACC` as proposed by
    `Forman 2006 <https://dl.acm.org/doi/abs/10.1145/1150402.1150423>`_ and
    `Forman 2008 <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_ that looks
    for the threshold that yields `tpr=1-fpr`.
    The goal is to bring improved stability to the denominator of the adjustment.

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`, defaults 5), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """

    def __init__(self, classifier: BaseEstimator=None, val_split=5):
        super().__init__(classifier, val_split)

    def condition(self, tpr, fpr) -> float:
        return abs(1 - (tpr + fpr))


class MS(ThresholdOptimization):
    """
    Median Sweep. Threshold Optimization variant for :class:`ACC` as proposed by
    `Forman 2006 <https://dl.acm.org/doi/abs/10.1145/1150402.1150423>`_ and
    `Forman 2008 <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_ that generates
    class prevalence estimates for all decision thresholds and returns the median of them all.
    The goal is to bring improved stability to the denominator of the adjustment.

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`, defaults 5), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """
    def __init__(self, classifier: BaseEstimator=None, val_split=5):
        super().__init__(classifier, val_split)

    def condition(self, tpr, fpr) -> float:
        return 1

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        decision_scores, y = classif_predictions.Xy
        # keeps all candidates
        tprs_fprs_thresholds = self._eval_candidate_thresholds(decision_scores, y)
        self.tprs = tprs_fprs_thresholds[:, 0]
        self.fprs = tprs_fprs_thresholds[:, 1]
        self.thresholds = tprs_fprs_thresholds[:, 2]
        return self

    def aggregate(self, classif_predictions: np.ndarray):
        prevalences = self.aggregate_with_threshold(classif_predictions, self.tprs, self.fprs, self.thresholds)
        if prevalences.ndim==2:
            prevalences = np.median(prevalences, axis=0)
        return prevalences


class MS2(MS):
    """
    Median Sweep 2. Threshold Optimization variant for :class:`ACC` as proposed by
    `Forman 2006 <https://dl.acm.org/doi/abs/10.1145/1150402.1150423>`_ and
    `Forman 2008 <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_ that generates
    class prevalence estimates for all decision thresholds and returns the median of for cases in
    which `tpr-fpr>0.25`
    The goal is to bring improved stability to the denominator of the adjustment.

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set in which the
        misclassification rates are to be estimated.
        This parameter can be indicated as a real value (between 0 and 1), representing a proportion of
        validation data, or as an integer, indicating that the misclassification rates should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`, defaults 5), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    """
    def __init__(self, classifier: BaseEstimator=None, val_split=5):
        super().__init__(classifier, val_split)

    def discard(self, tpr, fpr) -> bool:
        return (tpr-fpr) <= 0.25
