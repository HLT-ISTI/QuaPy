import numpy as np
import logging
from collections import namedtuple

from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import MultiLabelBinarizer

from NewMethods.fgsld.metrics import smoothmacroF1, isometric_brier_decomposition, isomerous_brier_decomposition

History = namedtuple('History', ('posteriors', 'priors', 'y', 'iteration', 'stopping_criterium'))
MeasureSingleHistory = namedtuple('MeasureSingleHistory', (
    'soft_acc', 'soft_f1', 'abs_errors', 'test_priors', 'train_priors', 'predict_priors', 'brier',
    'isometric_ref_loss', 'isometric_cal_loss', 'isomerous_ref_loss', 'isomerous_cal_loss'
))


def get_measures_single_history(history: History, multi_class) -> MeasureSingleHistory:
    y = history.y

    y_bin = MultiLabelBinarizer(classes=list(range(history.posteriors.shape[1]))).fit_transform(np.expand_dims(y, 1))

    soft_acc = soft_accuracy(y, history.posteriors)
    f1 = smoothmacroF1(y_bin, history.posteriors)

    if multi_class:
        test_priors = np.mean(y_bin, 0)
        abs_errors = abs(test_priors - history.priors)
        train_priors = history.priors
        predict_priors = np.mean(history.posteriors, 0)
        brier = 0
    else:
        test_priors = np.mean(y_bin, 0)[1]
        abs_errors = abs(test_priors - history.priors[1])
        train_priors = history.priors[1]
        predict_priors = np.mean(history.posteriors[:, 1])
        brier = brier_score_loss(y, history.posteriors[:, 1])

    isometric_cal_loss, isometric_ref_loss = isometric_brier_decomposition(y, history.posteriors)
    isomerous_em_cal_loss, isomerous_em_ref_loss = isomerous_brier_decomposition(y, history.posteriors)

    return MeasureSingleHistory(
        soft_acc, f1, abs_errors, test_priors, train_priors, predict_priors, brier, isometric_ref_loss,
        isometric_cal_loss, isomerous_em_ref_loss, isomerous_em_cal_loss
    )


def soft_accuracy(y, posteriors):
    return sum(posteriors[y == c][:, c].sum() for c in range(posteriors.shape[1])) / posteriors.sum()


def soft_f1(y, posteriors):
    cont_matrix = {
        'TPM': posteriors[y == 1][:, 1].sum(),
        'TNM': posteriors[y == 0][:, 0].sum(),
        'FPM': posteriors[y == 0][:, 1].sum(),
        'FNM': posteriors[y == 1][:, 0].sum()
    }
    precision = cont_matrix['TPM'] / (cont_matrix['TPM'] + cont_matrix['FPM'])
    recall = cont_matrix['TPM'] / (cont_matrix['TPM'] + cont_matrix['FNM'])
    return 2 * (precision * recall / (precision + recall))


def em(y, posteriors_zero, priors_zero, epsilon=1e-6, multi_class=False, return_posteriors_hist=False):
    """
    Implements the prior correction method based on EM presented in:
    "Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure"
    Saerens, Latinne and Decaestecker, 2002
    http://www.isys.ucl.ac.be/staff/marco/Publications/Saerens2002a.pdf

    :param y: true labels of test items, to measure accuracy, precision and recall.
    :param posteriors_zero: posterior probabilities on test items, as returned by a classifier. A 2D-array with shape
    Ø(items, classes).
    :param priors_zero: prior probabilities measured on training set.
    :param epsilon: stopping threshold.
    :param multi_class: whether the algorithm is running in a multi-label multi-class context or not.
    :param return_posteriors_hist: whether posteriors for each iteration should be returned or not. If true, the returned
    posteriors_s will actually be the list of posteriors for every iteration.
    :return: posteriors_s, priors_s, history: final adjusted posteriors, final adjusted priors, a list of length s
    where each element is a tuple with the step counter, the current priors (as list), the stopping criterium value,
    accuracy, precision and recall.
    """
    s = 0
    priors_s = np.copy(priors_zero)
    posteriors_s = np.copy(posteriors_zero)
    if return_posteriors_hist:
        posteriors_hist = [posteriors_s.copy()]
    val = 2 * epsilon
    history = list()
    history.append(get_measures_single_history(History(posteriors_zero, priors_zero, y, s, 1), multi_class))
    while not val < epsilon and s < 999:
        # M step
        priors_s_minus_one = priors_s.copy()
        priors_s = posteriors_s.mean(0)

        # E step
        ratios = priors_s / priors_zero
        denominators = 0
        for c in range(priors_zero.shape[0]):
            denominators += ratios[c] * posteriors_zero[:, c]
        for c in range(priors_zero.shape[0]):
            posteriors_s[:, c] = ratios[c] * posteriors_zero[:, c] / denominators

        # check for stop
        val = 0
        for i in range(len(priors_s_minus_one)):
            val += abs(priors_s_minus_one[i] - priors_s[i])

        logging.debug(f"Em iteration: {s}; Val: {val}")
        s += 1
        if return_posteriors_hist:
            posteriors_hist.append(posteriors_s.copy())
        history.append(get_measures_single_history(History(posteriors_s, priors_s, y, s, val), multi_class))

    if return_posteriors_hist:
        return posteriors_hist, priors_s, history
    return posteriors_s, priors_s, history
