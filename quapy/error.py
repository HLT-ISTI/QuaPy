import numpy as np
from sklearn.metrics import f1_score


def from_name(err_name):
    assert err_name in ERROR_NAMES, f'unknown error {err_name}'
    callable_error = globals()[err_name]
    if err_name in QUANTIFICATION_ERROR_SMOOTH_NAMES:
        eps = __check_eps()
        def bound_callable_error(y_true, y_pred):
            return callable_error(y_true, y_pred, eps)
        return bound_callable_error
    return callable_error


def f1e(y_true, y_pred):
    return 1. - f1_score(y_true, y_pred, average='macro')


def acce(y_true, y_pred):
    return 1. - (y_true == y_pred).mean()


def mae(prevs, prevs_hat):
    return ae(prevs, prevs_hat).mean()


def ae(p, p_hat):
    assert p.shape == p_hat.shape, f'wrong shape {p.shape} vs. {p_hat.shape}'
    return abs(p_hat-p).mean(axis=-1)


def mse(prevs, prevs_hat):
    return se(prevs, prevs_hat).mean()


def se(p, p_hat):
    return ((p_hat-p)**2).mean(axis=-1)


def mkld(prevs, prevs_hat, eps=None):
    return kld(prevs, prevs_hat, eps).mean()


def kld(p, p_hat, eps=None):
    eps = __check_eps(eps)
    sp = p+eps
    sp_hat = p_hat + eps
    return (sp*np.log(sp/sp_hat)).sum(axis=-1)


def mnkld(prevs, prevs_hat, eps=None):
    return nkld(prevs, prevs_hat, eps).mean()


def nkld(p, p_hat, eps=None):
    ekld = np.exp(kld(p, p_hat, eps))
    return 2. * ekld / (1 + ekld) - 1.


def mrae(p, p_hat, eps=None):
    return rae(p, p_hat, eps).mean()


def rae(p, p_hat, eps=None):
    eps = __check_eps(eps)
    p = smooth(p, eps)
    p_hat = smooth(p_hat, eps)
    return (abs(p-p_hat)/p).mean(axis=-1)


def smooth(p, eps):
    n_classes = p.shape[-1]
    return (p+eps)/(eps*n_classes + 1)


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
QUANTIFICATION_ERROR = {mae, mrae, mse, mkld, mnkld}
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

