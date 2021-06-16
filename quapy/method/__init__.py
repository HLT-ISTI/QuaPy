from . import aggregative
from . import base
from . import meta
from . import non_aggregative

EXPLICIT_LOSS_MINIMIZATION_METHODS = {
    aggregative.ELM,
    aggregative.SVMQ,
    aggregative.SVMAE,
    aggregative.SVMKLD,
    aggregative.SVMRAE,
    aggregative.SVMNKLD
}

AGGREGATIVE_METHODS = {
    aggregative.CC,
    aggregative.ACC,
    aggregative.PCC,
    aggregative.PACC,
    aggregative.EMQ,
    aggregative.HDy,
    aggregative.X,
    aggregative.T50,
    aggregative.MAX,
    aggregative.MS,
    aggregative.MS2,
} | EXPLICIT_LOSS_MINIMIZATION_METHODS


NON_AGGREGATIVE_METHODS = {
    non_aggregative.MaximumLikelihoodPrevalenceEstimation
}

META_METHODS = {
    meta.Ensemble,
    meta.QuaNet
}

QUANTIFICATION_METHODS = AGGREGATIVE_METHODS | NON_AGGREGATIVE_METHODS | META_METHODS



