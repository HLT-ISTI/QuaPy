"""Implementation of error measures used for quantification"""

import numpy as np
from sklearn.metrics import f1_score
import quapy as qp


def from_name(err_name):
    """Gets an error function from its name. E.g., `from_name("mae")`
    will return function :meth:`quapy.error.mae`

    :param err_name: string, the error name
    :return: a callable implementing the requested error
    """
    assert err_name in ERROR_NAMES, f'unknown error {err_name}'
    callable_error = globals()[err_name]
    return callable_error


def f1e(y_true, y_pred):
    """F1 error: simply computes the error in terms of macro :math:`F_1`, i.e.,
    :math:`1-F_1^M`, where :math:`F_1` is the harmonic mean of precision and recall,
    defined as :math:`\\frac{2tp}{2tp+fp+fn}`, with `tp`, `fp`, and `fn` standing
    for true positives, false positives, and false negatives, respectively.
    `Macro` averaging means the :math:`F_1` is computed for each category independently,
    and then averaged.

    :param y_true: array-like of true labels
    :param y_pred: array-like of predicted labels
    :return: :math:`1-F_1^M`
    """
    return 1. - f1_score(y_true, y_pred, average='macro')


def acce(y_true, y_pred):
    """Computes the error in terms of 1-accuracy. The accuracy is computed as
    :math:`\\frac{tp+tn}{tp+fp+fn+tn}`, with `tp`, `fp`, `fn`, and `tn` standing
    for true positives, false positives, false negatives, and true negatives,
    respectively

    :param y_true: array-like of true labels
    :param y_pred: array-like of predicted labels
    :return: 1-accuracy
    """
    return 1. - (y_true == y_pred).mean()


def mae(prevs, prevs_hat):
    """Computes the mean absolute error (see :meth:`quapy.error.ae`) across the sample pairs.

    :param prevs: array-like of shape `(n_samples, n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_samples, n_classes,)` with the predicted
        prevalence values
    :return: mean absolute error
    """
    return ae(prevs, prevs_hat).mean()


def ae(prevs, prevs_hat):
    """Computes the absolute error between the two prevalence vectors.
     Absolute error between two prevalence vectors :math:`p` and :math:`\\hat{p}`  is computed as
     :math:`AE(p,\\hat{p})=\\frac{1}{|\\mathcal{Y}|}\\sum_{y\\in \\mathcal{Y}}|\\hat{p}(y)-p(y)|`,
     where :math:`\\mathcal{Y}` are the classes of interest.

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :return: absolute error
    """
    assert prevs.shape == prevs_hat.shape, f'wrong shape {prevs.shape} vs. {prevs_hat.shape}'
    return abs(prevs_hat - prevs).mean(axis=-1)


def nae(prevs, prevs_hat):
    """Computes the normalized absolute error between the two prevalence vectors.
     Normalized absolute error between two prevalence vectors :math:`p` and :math:`\\hat{p}`  is computed as
     :math:`NAE(p,\\hat{p})=\\frac{AE(p,\\hat{p})}{z_{AE}}`,
     where :math:`z_{AE}=\\frac{2(1-\\min_{y\\in \\mathcal{Y}} p(y))}{|\\mathcal{Y}|}`, and :math:`\\mathcal{Y}`
     are the classes of interest.

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :return: normalized absolute error
    """
    assert prevs.shape == prevs_hat.shape, f'wrong shape {prevs.shape} vs. {prevs_hat.shape}'
    return abs(prevs_hat - prevs).sum(axis=-1)/(2*(1-prevs.min(axis=-1)))


def mnae(prevs, prevs_hat):
    """Computes the mean normalized absolute error (see :meth:`quapy.error.nae`) across the sample pairs.

    :param prevs: array-like of shape `(n_samples, n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_samples, n_classes,)` with the predicted
        prevalence values
    :return: mean normalized absolute error
    """
    return nae(prevs, prevs_hat).mean()


def mse(prevs, prevs_hat):
    """Computes the mean squared error (see :meth:`quapy.error.se`) across the sample pairs.

    :param prevs: array-like of shape `(n_samples, n_classes,)` with the
        true prevalence values
    :param prevs_hat: array-like of shape `(n_samples, n_classes,)` with the
        predicted prevalence values
    :return: mean squared error
    """
    return se(prevs, prevs_hat).mean()


def se(prevs, prevs_hat):
    """Computes the squared error between the two prevalence vectors.
     Squared error between two prevalence vectors :math:`p` and :math:`\\hat{p}`  is computed as
     :math:`SE(p,\\hat{p})=\\frac{1}{|\\mathcal{Y}|}\\sum_{y\\in \\mathcal{Y}}(\\hat{p}(y)-p(y))^2`,
     where
     :math:`\\mathcal{Y}` are the classes of interest.

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :return: absolute error
    """
    return ((prevs_hat - prevs) ** 2).mean(axis=-1)


def mkld(prevs, prevs_hat, eps=None):
    """Computes the mean Kullback-Leibler divergence (see :meth:`quapy.error.kld`) across the
    sample pairs. The distributions are smoothed using the `eps` factor
    (see :meth:`quapy.error.smooth`).

    :param prevs: array-like of shape `(n_samples, n_classes,)` with the true
        prevalence values
    :param prevs_hat: array-like of shape `(n_samples, n_classes,)` with the predicted
        prevalence values
    :param eps: smoothing factor. KLD is not defined in cases in which the distributions contain
        zeros; `eps` is typically set to be :math:`\\frac{1}{2T}`, with :math:`T` the sample size.
        If `eps=None`, the sample size will be taken from the environment variable `SAMPLE_SIZE`
        (which has thus to be set beforehand).
    :return: mean Kullback-Leibler distribution
    """
    return kld(prevs, prevs_hat, eps).mean()


def kld(prevs, prevs_hat, eps=None):
    """Computes the Kullback-Leibler divergence between the two prevalence distributions.
     Kullback-Leibler divergence between two prevalence distributions :math:`p` and :math:`\\hat{p}`
     is computed as
     :math:`KLD(p,\\hat{p})=D_{KL}(p||\\hat{p})=
     \\sum_{y\\in \\mathcal{Y}} p(y)\\log\\frac{p(y)}{\\hat{p}(y)}`,
     where :math:`\\mathcal{Y}` are the classes of interest.
     The distributions are smoothed using the `eps` factor (see :meth:`quapy.error.smooth`).

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :param eps: smoothing factor. KLD is not defined in cases in which the distributions contain
        zeros; `eps` is typically set to be :math:`\\frac{1}{2T}`, with :math:`T` the sample size.
        If `eps=None`, the sample size will be taken from the environment variable `SAMPLE_SIZE`
        (which has thus to be set beforehand).
    :return: Kullback-Leibler divergence between the two distributions
    """
    eps = __check_eps(eps)
    smooth_prevs = prevs + eps
    smooth_prevs_hat = prevs_hat + eps
    return (smooth_prevs*np.log(smooth_prevs/smooth_prevs_hat)).sum(axis=-1)


def mnkld(prevs, prevs_hat, eps=None):
    """Computes the mean Normalized Kullback-Leibler divergence (see :meth:`quapy.error.nkld`)
    across the sample pairs. The distributions are smoothed using the `eps` factor
    (see :meth:`quapy.error.smooth`).

    :param prevs: array-like of shape `(n_samples, n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_samples, n_classes,)` with the predicted
        prevalence values
    :param eps: smoothing factor. NKLD is not defined in cases in which the distributions contain
        zeros; `eps` is typically set to be :math:`\\frac{1}{2T}`, with :math:`T` the sample size.
        If `eps=None`, the sample size will be taken from the environment variable `SAMPLE_SIZE`
        (which has thus to be set beforehand).
    :return: mean Normalized Kullback-Leibler distribution
    """
    return nkld(prevs, prevs_hat, eps).mean()


def nkld(prevs, prevs_hat, eps=None):
    """Computes the Normalized Kullback-Leibler divergence between the two prevalence distributions.
     Normalized Kullback-Leibler divergence between two prevalence distributions :math:`p` and
     :math:`\\hat{p}` is computed as
     math:`NKLD(p,\\hat{p}) = 2\\frac{e^{KLD(p,\\hat{p})}}{e^{KLD(p,\\hat{p})}+1}-1`,
     where
     :math:`\\mathcal{Y}` are the classes of interest.
     The distributions are smoothed using the `eps` factor (see :meth:`quapy.error.smooth`).

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :param eps: smoothing factor. NKLD is not defined in cases in which the distributions
        contain zeros; `eps` is typically set to be :math:`\\frac{1}{2T}`, with :math:`T` the sample
        size. If `eps=None`, the sample size will be taken from the environment variable
        `SAMPLE_SIZE` (which has thus to be set beforehand).
    :return: Normalized Kullback-Leibler divergence between the two distributions
    """
    ekld = np.exp(kld(prevs, prevs_hat, eps))
    return 2. * ekld / (1 + ekld) - 1.


def mrae(prevs, prevs_hat, eps=None):
    """Computes the mean relative absolute error (see :meth:`quapy.error.rae`) across
    the sample pairs. The distributions are smoothed using the `eps` factor (see
    :meth:`quapy.error.smooth`).

    :param prevs: array-like of shape `(n_samples, n_classes,)` with the true
        prevalence values
    :param prevs_hat: array-like of shape `(n_samples, n_classes,)` with the predicted
        prevalence values
    :param eps: smoothing factor. `mrae` is not defined in cases in which the true
        distribution contains zeros; `eps` is typically set to be :math:`\\frac{1}{2T}`,
        with :math:`T` the sample size. If `eps=None`, the sample size will be taken from
        the environment variable `SAMPLE_SIZE` (which has thus to be set beforehand).
    :return: mean relative absolute error
    """
    return rae(prevs, prevs_hat, eps).mean()


def rae(prevs, prevs_hat, eps=None):
    """Computes the absolute relative error between the two prevalence vectors.
     Relative absolute error between two prevalence vectors :math:`p` and :math:`\\hat{p}`
     is computed as
     :math:`RAE(p,\\hat{p})=
     \\frac{1}{|\\mathcal{Y}|}\\sum_{y\\in \\mathcal{Y}}\\frac{|\\hat{p}(y)-p(y)|}{p(y)}`,
     where :math:`\\mathcal{Y}` are the classes of interest.
     The distributions are smoothed using the `eps` factor (see :meth:`quapy.error.smooth`).

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :param eps: smoothing factor. `rae` is not defined in cases in which the true distribution
        contains zeros; `eps` is typically set to be :math:`\\frac{1}{2T}`, with :math:`T` the
        sample size. If `eps=None`, the sample size will be taken from the environment variable
        `SAMPLE_SIZE` (which has thus to be set beforehand).
    :return: relative absolute error
    """
    eps = __check_eps(eps)
    prevs = smooth(prevs, eps)
    prevs_hat = smooth(prevs_hat, eps)
    return (abs(prevs - prevs_hat) / prevs).mean(axis=-1)


def nrae(prevs, prevs_hat, eps=None):
    """Computes the normalized absolute relative error between the two prevalence vectors.
     Relative absolute error between two prevalence vectors :math:`p` and :math:`\\hat{p}`
     is computed as
     :math:`NRAE(p,\\hat{p})= \\frac{RAE(p,\\hat{p})}{z_{RAE}}`,
     where
     :math:`z_{RAE} = \\frac{|\\mathcal{Y}|-1+\\frac{1-\\min_{y\\in \\mathcal{Y}} p(y)}{\\min_{y\\in \\mathcal{Y}} p(y)}}{|\\mathcal{Y}|}`
     and :math:`\\mathcal{Y}` are the classes of interest.
     The distributions are smoothed using the `eps` factor (see :meth:`quapy.error.smooth`).

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :param eps: smoothing factor. `nrae` is not defined in cases in which the true distribution
        contains zeros; `eps` is typically set to be :math:`\\frac{1}{2T}`, with :math:`T` the
        sample size. If `eps=None`, the sample size will be taken from the environment variable
        `SAMPLE_SIZE` (which has thus to be set beforehand).
    :return: normalized relative absolute error
    """
    eps = __check_eps(eps)
    prevs = smooth(prevs, eps)
    prevs_hat = smooth(prevs_hat, eps)
    min_p = prevs.min(axis=-1)
    return (abs(prevs - prevs_hat) / prevs).sum(axis=-1)/(prevs.shape[-1]-1+(1-min_p)/min_p)


def mnrae(prevs, prevs_hat, eps=None):
    """Computes the mean normalized relative absolute error (see :meth:`quapy.error.nrae`) across
    the sample pairs. The distributions are smoothed using the `eps` factor (see
    :meth:`quapy.error.smooth`).

    :param prevs: array-like of shape `(n_samples, n_classes,)` with the true
        prevalence values
    :param prevs_hat: array-like of shape `(n_samples, n_classes,)` with the predicted
        prevalence values
    :param eps: smoothing factor. `mnrae` is not defined in cases in which the true
        distribution contains zeros; `eps` is typically set to be :math:`\\frac{1}{2T}`,
        with :math:`T` the sample size. If `eps=None`, the sample size will be taken from
        the environment variable `SAMPLE_SIZE` (which has thus to be set beforehand).
    :return: mean normalized relative absolute error
    """
    return nrae(prevs, prevs_hat, eps).mean()


def smooth(prevs, eps):
    """ Smooths a prevalence distribution with :math:`\\epsilon` (`eps`) as:
    :math:`\\underline{p}(y)=\\frac{\\epsilon+p(y)}{\\epsilon|\\mathcal{Y}|+
    \\displaystyle\\sum_{y\\in \\mathcal{Y}}p(y)}`

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param eps: smoothing factor
    :return: array-like of shape `(n_classes,)` with the smoothed distribution
    """
    n_classes = prevs.shape[-1]
    return (prevs + eps) / (eps * n_classes + 1)


def __check_eps(eps=None):
    if eps is None:
        sample_size = qp.environ['SAMPLE_SIZE']
        if sample_size is None:
            raise ValueError('eps was not defined, and qp.environ["SAMPLE_SIZE"] was not set')
        eps = 1. / (2. * sample_size)
    return eps


CLASSIFICATION_ERROR = {f1e, acce}
QUANTIFICATION_ERROR = {mae, mnae, mrae, mnrae, mse, mkld, mnkld}
QUANTIFICATION_ERROR_SINGLE = {ae, nae, rae, nrae, se, kld, nkld}
QUANTIFICATION_ERROR_SMOOTH = {kld, nkld, rae, nrae, mkld, mnkld, mrae}
CLASSIFICATION_ERROR_NAMES = {func.__name__ for func in CLASSIFICATION_ERROR}
QUANTIFICATION_ERROR_NAMES = {func.__name__ for func in QUANTIFICATION_ERROR}
QUANTIFICATION_ERROR_SINGLE_NAMES = {func.__name__ for func in QUANTIFICATION_ERROR_SINGLE}
QUANTIFICATION_ERROR_SMOOTH_NAMES = {func.__name__ for func in QUANTIFICATION_ERROR_SMOOTH}
ERROR_NAMES = \
    CLASSIFICATION_ERROR_NAMES | QUANTIFICATION_ERROR_NAMES | QUANTIFICATION_ERROR_SINGLE_NAMES

f1_error = f1e
acc_error = acce
mean_absolute_error = mae
absolute_error = ae
mean_relative_absolute_error = mrae
relative_absolute_error = rae
normalized_absolute_error = nae
normalized_relative_absolute_error = nrae
mean_normalized_absolute_error = mnae
mean_normalized_relative_absolute_error = mnrae
