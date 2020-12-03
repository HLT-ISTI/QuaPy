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


