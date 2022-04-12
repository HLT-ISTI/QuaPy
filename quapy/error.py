import quapy as qp
import numpy as np
from sklearn.metrics import f1_score


def from_name(err_name):
    """Gets an error function from its name. E.g., `from_name("mae")` will return function :meth:`quapy.error.mae`

    :param err_name: string, the error name
    :return: a callable implementing the requested error
    """
    assert err_name in ERROR_NAMES, f'unknown error {err_name}'
    callable_error = globals()[err_name]
    if err_name in QUANTIFICATION_ERROR_SMOOTH_NAMES:
        eps = __check_eps()
        def bound_callable_error(y_true, y_pred):
            return callable_error(y_true, y_pred, eps)
        return bound_callable_error
    return callable_error


def f1e(y_true, y_pred):
    """F1 error: simply computes the error in terms of macro :math:`F_1`, i.e., :math:`1-F_1^M`,
    where :math:`F_1` is the harmonic mean of precision and recall, defined as :math:`\\frac{2tp}{2tp+fp+fn}`,
    with `tp`, `fp`, and `fn` standing for true positives, false positives, and false negatives, respectively.
    `Macro` averaging means the :math:`F_1` is computed for each category independently, and then averaged.

    :param y_true: array-like of true labels
    :param y_pred: array-like of predicted labels
    :return: :math:`1-F_1^M`
    """
    return 1. - f1_score(y_true, y_pred, average='macro')


def acce(y_true, y_pred):
    """Computes the error in terms of 1-accuracy. The accuracy is computed as :math:`\\frac{tp+tn}{tp+fp+fn+tn}`, with
    `tp`, `fp`, `fn`, and `tn` standing for true positives, false positives, false negatives, and true negatives,
    respectively

    :param y_true: array-like of true labels
    :param y_pred: array-like of predicted labels
    :return: 1-accuracy
    """
    return 1. - (y_true == y_pred).mean()


def mae(prevs, prevs_hat):
    """Computes the mean absolute error (see :meth:`quapy.error.ae`) across the sample pairs.

    :param prevs: array-like of shape `(n_samples, n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_samples, n_classes,)` with the predicted prevalence values
    :return: mean absolute error
    """
    return ae(prevs, prevs_hat).mean()


def ae(prevs, prevs_hat):
    """Computes the absolute error between the two prevalence vectors.
     Absolute error between two prevalence vectors :math:`p` and :math:`\\hat{p}`  is computed as
     :math:`AE(p,\\hat{p})=\\frac{1}{|\\mathcal{Y}|}\\sum_{y\in \mathcal{Y}}|\\hat{p}(y)-p(y)|`,
     where :math:`\\mathcal{Y}` are the classes of interest.

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :return: absolute error
    """
    assert prevs.shape == prevs_hat.shape, f'wrong shape {prevs.shape} vs. {prevs_hat.shape}'
    return abs(prevs_hat - prevs).mean(axis=-1)


def mse(prevs, prevs_hat):
    """Computes the mean squared error (see :meth:`quapy.error.se`) across the sample pairs.

    :param prevs: array-like of shape `(n_samples, n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_samples, n_classes,)` with the predicted prevalence values
    :return: mean squared error
    """
    return se(prevs, prevs_hat).mean()


def se(p, p_hat):
    """Computes the squared error between the two prevalence vectors.
     Squared error between two prevalence vectors :math:`p` and :math:`\\hat{p}`  is computed as
     :math:`SE(p,\\hat{p})=\\frac{1}{|\\mathcal{Y}|}\\sum_{y\in \mathcal{Y}}(\\hat{p}(y)-p(y))^2`, where
     :math:`\\mathcal{Y}` are the classes of interest.

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :return: absolute error
    """
    return ((p_hat-p)**2).mean(axis=-1)


def mkld(prevs, prevs_hat, eps=None):
    """Computes the mean Kullback-Leibler divergence (see :meth:`quapy.error.kld`) across the sample pairs.
    The distributions are smoothed using the `eps` factor (see :meth:`quapy.error.smooth`).

    :param prevs: array-like of shape `(n_samples, n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_samples, n_classes,)` with the predicted prevalence values
    :param eps: smoothing factor. KLD is not defined in cases in which the distributions contain zeros; `eps`
        is typically set to be :math:`\\frac{1}{2T}`, with :math:`T` the sample size. If `eps=None`, the sample size
        will be taken from the environment variable `SAMPLE_SIZE` (which has thus to be set beforehand).
    :return: mean Kullback-Leibler distribution
    """
    return kld(prevs, prevs_hat, eps).mean()


def kld(p, p_hat, eps=None):
    """Computes the Kullback-Leibler divergence between the two prevalence distributions.
     Kullback-Leibler divergence between two prevalence distributions :math:`p` and :math:`\\hat{p}` is computed as
     :math:`KLD(p,\\hat{p})=D_{KL}(p||\\hat{p})=\\sum_{y\\in \\mathcal{Y}} p(y)\\log\\frac{p(y)}{\\hat{p}(y)}`, where
     :math:`\\mathcal{Y}` are the classes of interest.
     The distributions are smoothed using the `eps` factor (see :meth:`quapy.error.smooth`).

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :param eps: smoothing factor. KLD is not defined in cases in which the distributions contain zeros; `eps`
        is typically set to be :math:`\\frac{1}{2T}`, with :math:`T` the sample size. If `eps=None`, the sample size
        will be taken from the environment variable `SAMPLE_SIZE` (which has thus to be set beforehand).
    :return: Kullback-Leibler divergence between the two distributions
    """
    eps = __check_eps(eps)
    sp = p+eps
    sp_hat = p_hat + eps
    return (sp*np.log(sp/sp_hat)).sum(axis=-1)


def mnkld(prevs, prevs_hat, eps=None):
    """Computes the mean Normalized Kullback-Leibler divergence (see :meth:`quapy.error.nkld`) across the sample pairs.
    The distributions are smoothed using the `eps` factor (see :meth:`quapy.error.smooth`).

    :param prevs: array-like of shape `(n_samples, n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_samples, n_classes,)` with the predicted prevalence values
    :param eps: smoothing factor. NKLD is not defined in cases in which the distributions contain zeros; `eps`
        is typically set to be :math:`\\frac{1}{2T}`, with :math:`T` the sample size. If `eps=None`, the sample size
        will be taken from the environment variable `SAMPLE_SIZE` (which has thus to be set beforehand).
    :return: mean Normalized Kullback-Leibler distribution
    """
    return nkld(prevs, prevs_hat, eps).mean()


def nkld(p, p_hat, eps=None):
    """Computes the Normalized Kullback-Leibler divergence between the two prevalence distributions.
     Normalized Kullback-Leibler divergence between two prevalence distributions :math:`p` and :math:`\\hat{p}`
     is computed as :math:`NKLD(p,\\hat{p}) = 2\\frac{e^{KLD(p,\\hat{p})}}{e^{KLD(p,\\hat{p})}+1}-1`, where
     :math:`\\mathcal{Y}` are the classes of interest.
     The distributions are smoothed using the `eps` factor (see :meth:`quapy.error.smooth`).

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :param eps: smoothing factor. NKLD is not defined in cases in which the distributions contain zeros; `eps`
        is typically set to be :math:`\\frac{1}{2T}`, with :math:`T` the sample size. If `eps=None`, the sample size
        will be taken from the environment variable `SAMPLE_SIZE` (which has thus to be set beforehand).
    :return: Normalized Kullback-Leibler divergence between the two distributions
    """
    ekld = np.exp(kld(p, p_hat, eps))
    return 2. * ekld / (1 + ekld) - 1.


def mrae(p, p_hat, eps=None):
    """Computes the mean relative absolute error (see :meth:`quapy.error.rae`) across the sample pairs.
    The distributions are smoothed using the `eps` factor (see :meth:`quapy.error.smooth`).

    :param prevs: array-like of shape `(n_samples, n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_samples, n_classes,)` with the predicted prevalence values
    :param eps: smoothing factor. `mrae` is not defined in cases in which the true distribution contains zeros; `eps`
        is typically set to be :math:`\\frac{1}{2T}`, with :math:`T` the sample size. If `eps=None`, the sample size
        will be taken from the environment variable `SAMPLE_SIZE` (which has thus to be set beforehand).
    :return: mean relative absolute error
    """
    return rae(p, p_hat, eps).mean()


def rae(p, p_hat, eps=None):
    """Computes the absolute relative error between the two prevalence vectors.
     Relative absolute error between two prevalence vectors :math:`p` and :math:`\\hat{p}`  is computed as
     :math:`RAE(p,\\hat{p})=\\frac{1}{|\\mathcal{Y}|}\\sum_{y\in \mathcal{Y}}\\frac{|\\hat{p}(y)-p(y)|}{p(y)}`,
     where :math:`\\mathcal{Y}` are the classes of interest.
     The distributions are smoothed using the `eps` factor (see :meth:`quapy.error.smooth`).

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :param eps: smoothing factor. `rae` is not defined in cases in which the true distribution contains zeros; `eps`
        is typically set to be :math:`\\frac{1}{2T}`, with :math:`T` the sample size. If `eps=None`, the sample size
        will be taken from the environment variable `SAMPLE_SIZE` (which has thus to be set beforehand).
    :return: relative absolute error
    """
    eps = __check_eps(eps)
    p = smooth(p, eps)
    p_hat = smooth(p_hat, eps)
    return (abs(p-p_hat)/p).mean(axis=-1)


def smooth(prevs, eps):
    """ Smooths a prevalence distribution with :math:`\epsilon` (`eps`) as:
    :math:`\\underline{p}(y)=\\frac{\\epsilon+p(y)}{\\epsilon|\\mathcal{Y}|+\\displaystyle\\sum_{y\\in \\mathcal{Y}}p(y)}`

    :param prevs: array-like of shape `(n_classes,)` with the true prevalence values
    :param eps: smoothing factor
    :return: array-like of shape `(n_classes,)` with the smoothed distribution
    """
    n_classes = prevs.shape[-1]
    return (prevs + eps) / (eps * n_classes + 1)


def __check_eps(eps=None):
    if eps is None:
        import quapy as qp
        sample_size = qp.environ['SAMPLE_SIZE']
        if sample_size is None:
            raise ValueError('eps was not defined, and qp.environ["SAMPLE_SIZE"] was not set')
        else:
            eps = 1. / (2. * sample_size)
    return eps


CLASSIFICATION_ERROR = {f1e, acce}
QUANTIFICATION_ERROR = {mae, mrae, mse, mkld, mnkld, ae, rae, se, kld, nkld}
QUANTIFICATION_ERROR_SMOOTH = {kld, nkld, rae, mkld, mnkld, mrae}
CLASSIFICATION_ERROR_NAMES = {func.__name__ for func in CLASSIFICATION_ERROR}
QUANTIFICATION_ERROR_NAMES = {func.__name__ for func in QUANTIFICATION_ERROR}
QUANTIFICATION_ERROR_SMOOTH_NAMES = {func.__name__ for func in QUANTIFICATION_ERROR_SMOOTH}
ERROR_NAMES = CLASSIFICATION_ERROR_NAMES | QUANTIFICATION_ERROR_NAMES

f1_error = f1e
acc_error = acce
mean_absolute_error = mae
absolute_error = ae
mean_relative_absolute_error = mrae
relative_absolute_error = rae

