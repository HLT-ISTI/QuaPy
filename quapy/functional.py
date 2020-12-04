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
    unique, counts = np.unique(labels, return_counts=True)
    by_class = defaultdict(lambda:0, dict(zip(unique, counts)))
    prevalences = np.asarray([by_class[ci] for ci in range(n_classes)], dtype=np.float)
    prevalences /= prevalences.sum()
    return prevalences


def prevalence_from_probabilities(posteriors, binarize: bool = False):
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


