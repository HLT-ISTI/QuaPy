from . import base
from . import aggregative
from . import non_aggregative
from . import meta


AGGREGATIVE_METHODS = {
    aggregative.CC,
    aggregative.ACC,
    aggregative.PCC,
    aggregative.PACC,
    aggregative.ELM,
    aggregative.EMQ,
    aggregative.HDy
}

NON_AGGREGATIVE_METHODS = {
    non_aggregative.MaximumLikelihoodPrevalenceEstimation
}

META_METHODS = {
    meta.QuaNet
}

QUANTIFICATION_METHODS = AGGREGATIVE_METHODS | NON_AGGREGATIVE_METHODS | META_METHODS



