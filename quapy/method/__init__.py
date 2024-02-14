from . import base
from . import aggregative
from . import non_aggregative
from . import meta

AGGREGATIVE_METHODS = {
    aggregative.CC,
    aggregative.ACC,
    aggregative.PCC,
    aggregative.PACC,
    aggregative.EMQ,
    aggregative.HDy,
    aggregative.DyS,
    aggregative.SMM,
    aggregative.X,
    aggregative.T50,
    aggregative.MAX,
    aggregative.MS,
    aggregative.MS2,
    aggregative.KDEyML,
    aggregative.KDEyCS,
    aggregative.KDEyHD,
}


NON_AGGREGATIVE_METHODS = {
    non_aggregative.MaximumLikelihoodPrevalenceEstimation
}

META_METHODS = {
    meta.Ensemble,
    meta.QuaNet
}

QUANTIFICATION_METHODS = AGGREGATIVE_METHODS | NON_AGGREGATIVE_METHODS | META_METHODS



