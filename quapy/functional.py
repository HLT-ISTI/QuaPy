import itertools
from collections import defaultdict
import scipy
import numpy as np


def artificial_prevalence_sampling(dimensions, n_prevalences=21, repeat=1, return_constrained_dim=False):
    """
    Generates vectors of prevalence values artificially drawn from an exhaustive grid of prevalence values. The
    number of prevalence values explored for each dimension depends on `n_prevalences`, so that, if, for example,
    `n_prevalences=11` then the prevalence values of the grid are taken from [0, 0.1, 0.2, ..., 0.9, 1]. Only
    valid prevalence distributions are returned, i.e., vectors of prevalence values that sum up to 1. For each
    valid vector of prevalence values, `repeat` copies are returned. The vector of prevalence values can be
    implicit (by setting `return_constrained_dim=False`), meaning that the last dimension (which is constrained
    to 1 - sum of the rest) is not returned (note that, quite obviously, in this case the vector does not sum up to 1).

    :param dimensions: the number of classes
    :param n_prevalences: the number of equidistant prevalence points to extract from the [0,1] interval for the grid
        (default is 21)
    :param repeat: number of copies for each valid prevalence vector (default is 1)
    :param return_constrained_dim: set to True to return all dimensions, or to False (default) for ommitting the
        constrained dimension
    :return: a `np.ndarray` of shape `(n, dimensions)` if `return_constrained_dim=True` or of shape `(n, dimensions-1)`
        if `return_constrained_dim=False`, where `n` is the number of valid combinations found in the grid multiplied
        by `repeat`
    """
    s = np.linspace(0., 1., n_prevalences, endpoint=True)
    s = [s] * (dimensions - 1)
    prevs = [p for p in itertools.product(*s, repeat=1) if sum(p)<=1]
    if return_constrained_dim:
        prevs = [p+(1-sum(p),) for p in prevs]
    prevs = np.asarray(prevs).reshape(len(prevs), -1)
    if repeat>1:
        prevs = np.repeat(prevs, repeat, axis=0)
    return prevs


def prevalence_linspace(n_prevalences=21, repeats=1, smooth_limits_epsilon=0.01):
    """
    Produces an array of uniformly separated values of prevalence.
    By default, produces an array of 21 prevalence values, with
    step 0.05 and with the limits smoothed, i.e.:
    [0.01, 0.05, 0.10, 0.15, ..., 0.90, 0.95, 0.99]

    :param n_prevalences: the number of prevalence values to sample from the [0,1] interval (default 21)
    :param repeats: number of times each prevalence is to be repeated (defaults to 1)
    :param smooth_limits_epsilon: the quantity to add and subtract to the limits 0 and 1
    :return: an array of uniformly separated prevalence values
    """
    p = np.linspace(0., 1., num=n_prevalences, endpoint=True)
    p[0] += smooth_limits_epsilon
    p[-1] -= smooth_limits_epsilon
    if p[0] > p[1]:
        raise ValueError(f'the smoothing in the limits is greater than the prevalence step')
    if repeats > 1:
        p = np.repeat(p, repeats)
    return p


def prevalence_from_labels(labels, classes):
    """
    Computed the prevalence values from a vector of labels.

    :param labels: array-like of shape `(n_instances)` with the label for each instance
    :param classes: the class labels. This is needed in order to correctly compute the prevalence vector even when
        some classes have no examples.
    :return: an ndarray of shape `(len(classes))` with the class prevalence values
    """
    if labels.ndim != 1:
        raise ValueError(f'param labels does not seem to be a ndarray of label predictions')
    unique, counts = np.unique(labels, return_counts=True)
    by_class = defaultdict(lambda:0, dict(zip(unique, counts)))
    prevalences = np.asarray([by_class[class_] for class_ in classes], dtype=np.float)
    prevalences /= prevalences.sum()
    return prevalences


def prevalence_from_probabilities(posteriors, binarize: bool = False):
    """
    Returns a vector of prevalence values from a matrix of posterior probabilities.

    :param posteriors: array-like of shape `(n_instances, n_classes,)` with posterior probabilities for each class
    :param binarize: set to True (default is False) for computing the prevalence values on crisp decisions (i.e.,
        converting the vectors of posterior probabilities into class indices, by taking the argmax).
    :return: array of shape `(n_classes,)` containing the prevalence values
    """
    if posteriors.ndim != 2:
        raise ValueError(f'param posteriors does not seem to be a ndarray of posteior probabilities')
    if binarize:
        predictions = np.argmax(posteriors, axis=-1)
        return prevalence_from_labels(predictions, np.arange(posteriors.shape[1]))
    else:
        prevalences = posteriors.mean(axis=0)
        prevalences /= prevalences.sum()
        return prevalences


def HellingerDistance(P, Q):
    """
    Computes the Hellingher Distance (HD) between (discretized) distributions `P` and `Q`.
    The HD for two discrete distributions of `k` bins is defined as:

    .. math::
        HD(P,Q) = \\frac{ 1 }{ \\sqrt{ 2 } } \\sqrt{ \sum_{i=1}^k ( \\sqrt{p_i} - \\sqrt{q_i} )^2 }

    :param P: real-valued array-like of shape `(k,)` representing a discrete distribution
    :param Q: real-valued array-like of shape `(k,)` representing a discrete distribution
    :return: float
    """
    return np.sqrt(np.sum((np.sqrt(P) - np.sqrt(Q))**2))


def uniform_prevalence_sampling(n_classes, size=1):
    """
    Implements the `Kraemer algorithm <http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf>`_
    for sampling uniformly at random from the unit simplex. This implementation is adapted from this
    `post <https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex>_`.

    :param n_classes: integer, number of classes (dimensionality of the simplex)
    :param size: number of samples to return
    :return: `np.ndarray` of shape `(size, n_classes,)` if `size>1`, or of shape `(n_classes,)` otherwise
    """
    if n_classes == 2:
        u = np.random.rand(size)
        u = np.vstack([1-u, u]).T
    else:
        u = np.random.rand(size, n_classes-1)
        u.sort(axis=-1)
        _0s = np.zeros(shape=(size, 1))
        _1s = np.ones(shape=(size, 1))
        a = np.hstack([_0s, u])
        b = np.hstack([u, _1s])
        u = b-a
    if size == 1:
        u = u.flatten()
    return u


uniform_simplex_sampling = uniform_prevalence_sampling


def strprev(prevalences, prec=3):
    """
    Returns a string representation for a prevalence vector. E.g.,

    >>> strprev([1/3, 2/3], prec=2)
    >>> '[0.33, 0.67]'

    :param prevalences: a vector of prevalence values
    :param prec: float precision
    :return: string
    """
    return '['+ ', '.join([f'{p:.{prec}f}' for p in prevalences]) + ']'


def adjusted_quantification(prevalence_estim, tpr, fpr, clip=True):
    """
    Implements the adjustment of ACC and PACC for the binary case. The adjustment for a prevalence estimate of the
    positive class `p` comes down to computing:

    .. math::
        ACC(p) = \\frac{ p - fpr }{ tpr - fpr }


    :param prevalence_estim: float, the estimated value for the positive class
    :param tpr: float, the true positive rate of the classifier
    :param fpr: float, the false positive rate of the classifier
    :param clip: set to True (default) to clip values that might exceed the range [0,1]
    :return: float, the adjusted count
    """

    den = tpr - fpr
    if den == 0:
        den += 1e-8
    adjusted = (prevalence_estim - fpr) / den
    if clip:
        adjusted = np.clip(adjusted, 0., 1.)
    return adjusted


def normalize_prevalence(prevalences):
    """
    Normalize a vector or matrix of prevalence values. The normalization consists of applying a L1 normalization in
    cases in which the prevalence values are not all-zeros, and to convert the prevalence values into `1/n_classes` in
    cases in which all values are zero.

    :param prevalences: array-like of shape `(n_classes,)` or of shape `(n_samples, n_classes,)` with prevalence values
    :return: a normalized vector or matrix of prevalence values
    """
    prevalences = np.asarray(prevalences)
    n_classes = prevalences.shape[-1]
    accum = prevalences.sum(axis=-1, keepdims=True)
    prevalences = np.true_divide(prevalences, accum, where=accum>0)
    allzeros = accum.flatten()==0
    if any(allzeros):
        if prevalences.ndim == 1:
            prevalences = np.full(shape=n_classes, fill_value=1./n_classes)
        else:
            prevalences[accum.flatten()==0] = np.full(shape=n_classes, fill_value=1./n_classes)
    return prevalences


def __num_prevalence_combinations_depr(n_prevpoints:int, n_classes:int, n_repeats:int=1):
    """
    Computes the number of prevalence combinations in the n_classes-dimensional simplex if `nprevpoints` equally distant
    prevalence values are generated and `n_repeats` repetitions are requested.

    :param n_classes: integer, number of classes
    :param n_prevpoints: integer, number of prevalence points.
    :param n_repeats: integer, number of repetitions for each prevalence combination
    :return: The number of possible combinations. For example, if n_classes=2, n_prevpoints=5, n_repeats=1, then the
    number of possible combinations are 5, i.e.: [0,1], [0.25,0.75], [0.50,0.50], [0.75,0.25], and [1.0,0.0]
    """
    __cache={}
    def __f(nc,np):
        if (nc,np) in __cache:  # cached result
            return __cache[(nc,np)]
        if nc==1:  # stop condition
            return 1
        else:  # recursive call
            x = sum([__f(nc-1, np-i) for i in range(np)])
            __cache[(nc,np)] = x
            return x
    return __f(n_classes, n_prevpoints) * n_repeats


def num_prevalence_combinations(n_prevpoints:int, n_classes:int, n_repeats:int=1):
    """
    Computes the number of valid prevalence combinations in the n_classes-dimensional simplex if `n_prevpoints` equally
    distant prevalence values are generated and `n_repeats` repetitions are requested.
    The computation comes down to calculating:

    .. math::
        \\binom{N+C-1}{C-1} \\times r

    where `N` is `n_prevpoints-1`, i.e., the number of probability mass blocks to allocate, `C` is the number of
    classes, and `r` is `n_repeats`. This solution comes from the
    `Stars and Bars <https://brilliant.org/wiki/integer-equations-star-and-bars/>`_ problem.

    :param n_classes: integer, number of classes
    :param n_prevpoints: integer, number of prevalence points.
    :param n_repeats: integer, number of repetitions for each prevalence combination
    :return: The number of possible combinations. For example, if n_classes=2, n_prevpoints=5, n_repeats=1, then the
    number of possible combinations are 5, i.e.: [0,1], [0.25,0.75], [0.50,0.50], [0.75,0.25], and [1.0,0.0]
    """
    N = n_prevpoints-1
    C = n_classes
    r = n_repeats
    return int(scipy.special.binom(N + C - 1, C - 1) * r)


def get_nprevpoints_approximation(combinations_budget:int, n_classes:int, n_repeats:int=1):
    """
    Searches for the largest number of (equidistant) prevalence points to define for each of the `n_classes` classes so
    that the number of valid prevalence values generated as combinations of prevalence points (points in a
    `n_classes`-dimensional simplex) do not exceed combinations_budget.

    :param combinations_budget: integer, maximum number of combinatios allowed
    :param n_classes: integer, number of classes
    :param n_repeats: integer, number of repetitions for each prevalence combination
    :return: the largest number of prevalence points that generate less than combinations_budget valid prevalences
    """
    assert n_classes > 0 and n_repeats > 0 and combinations_budget > 0, 'parameters must be positive integers'
    n_prevpoints = 1
    while True:
        combinations = num_prevalence_combinations(n_prevpoints, n_classes, n_repeats)
        if combinations > combinations_budget:
            return n_prevpoints-1
        else:
            n_prevpoints += 1

