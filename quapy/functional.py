from collections import defaultdict
import numpy as np
import itertools


def artificial_prevalence_sampling(dimensions, n_prevalences=21, repeat=1, return_constrained_dim=False):
    s = np.linspace(0., 1., n_prevalences, endpoint=True)
    s = [s] * (dimensions - 1)
    prevs = [p for p in itertools.product(*s, repeat=1) if sum(p)<=1]
    if return_constrained_dim:
        prevs = [p+(1-sum(p),) for p in prevs]
    prevs = np.asarray(prevs).reshape(len(prevs), -1)
    if repeat>1:
        prevs = np.repeat(prevs, repeat, axis=0)
    return prevs


def prevalence_linspace(n_prevalences=21, repeat=1, smooth_limits_epsilon=0.01):
    """
    Produces a uniformly separated values of prevalence. By default, produces an array 21 prevalences, with step 0.05
    and with the limits smoothed, i.e.:
    [0.01, 0.05, 0.10, 0.15, ..., 0.90, 0.95, 0.99]
    :param n_prevalences: the number of prevalence values to sample from the [0,1] interval (default 21)
    :param repeat: number of times each prevalence is to be repeated (defaults to 1)
    :param smooth_limits_epsilon: the quantity to add and subtract to the limits 0 and 1
    :return: an array of uniformly separated prevalence values
    """
    p = np.linspace(0., 1., num=n_prevalences, endpoint=True)
    p[0] += smooth_limits_epsilon
    p[-1] -= smooth_limits_epsilon
    if p[0] > p[1]:
        raise ValueError(f'the smoothing in the limits is greater than the prevalence step')
    if repeat > 1:
        p = np.repeat(p, repeat)
    return p


def prevalence_from_labels(labels, n_classes):
    if labels.ndim != 1:
        raise ValueError(f'param labels does not seem to be a ndarray of label predictions')
    unique, counts = np.unique(labels, return_counts=True)
    by_class = defaultdict(lambda:0, dict(zip(unique, counts)))
    prevalences = np.asarray([by_class[ci] for ci in range(n_classes)], dtype=np.float)
    prevalences /= prevalences.sum()
    return prevalences


def prevalence_from_probabilities(posteriors, binarize: bool = False):
    if posteriors.ndim != 2:
        raise ValueError(f'param posteriors does not seem to be a ndarray of posteior probabilities')
    if binarize:
        predictions = np.argmax(posteriors, axis=-1)
        return prevalence_from_labels(predictions, n_classes=posteriors.shape[1])
    else:
        prevalences = posteriors.mean(axis=0)
        prevalences /= prevalences.sum()
        return prevalences


def strprev(prevalences, prec=3):
    return '['+ ', '.join([f'{p:.{prec}f}' for p in prevalences]) + ']'


def adjusted_quantification(prevalence_estim, tpr, fpr, clip=True):
    den = tpr - fpr
    if den == 0:
        den += 1e-8
    adjusted = (prevalence_estim - fpr) / den
    if clip:
        adjusted = np.clip(adjusted, 0., 1.)
    return adjusted


def normalize_prevalence(prevalences):
    assert prevalences.ndim==1, 'unexpected shape'
    accum = prevalences.sum()
    if accum > 0:
        return prevalences / accum
    else:
        # if all classifiers are trivial rejectors
        return np.ones_like(prevalences) / prevalences.size



def num_prevalence_combinations(n_prevpoints:int, n_classes:int, n_repeats:int=1):
    """
    Computes the number of prevalence combinations in the n_classes-dimensional simplex if nprevpoints equally distant
    prevalences are generated and n_repeats repetitions are requested
    :param n_classes: number of classes
    :param n_prevpoints: number of prevalence points.
    :param n_repeats: number of repetitions for each prevalence combination
    :return: The number of possible combinations. For example, if n_classes=2, n_prevpoints=5, n_repeats=1, then the
    number of possible combinations are 5, i.e.: [0,1], [0.25,0.75], [0.50,0.50], [0.75,0.25], and [1.0,0.0]
    """
    __cache={}
    def __f(nc,np):
        if (nc,np) in __cache:
            return __cache[(nc,np)]
        if nc==1:
            return 1
        else:
            x = sum([__f(nc-1, np-i) for i in range(np)])
            __cache[(nc,np)] = x
            return x
    return __f(n_classes, n_prevpoints) * n_repeats


def get_nprevpoints_approximation(combinations_budget:int, n_classes:int, n_repeats:int=1):
    """
    Searches for the largest number of (equidistant) prevalence points to define for each of the n_classes classes so that
    the number of valid prevalences generated as combinations of prevalence points (points in a n_classes-dimensional
    simplex) do not exceed combinations_budget.
    :param n_classes: number of classes
    :param n_repeats: number of repetitions for each prevalence combination
    :param combinations_budget: maximum number of combinatios allowed
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

