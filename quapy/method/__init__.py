from . import base
from . import aggregative
from . import non_aggregative
from . import meta


AGGREGATIVE_METHODS = {
    aggregative.ClassifyAndCount,
    aggregative.AdjustedClassifyAndCount,
    aggregative.ProbabilisticClassifyAndCount,
    aggregative.ProbabilisticAdjustedClassifyAndCount,
    aggregative.ExplicitLossMinimisation,
    aggregative.ExpectationMaximizationQuantifier,
    aggregative.HellingerDistanceY
}

NON_AGGREGATIVE_METHODS = {
    non_aggregative.MaximumLikelihoodPrevalenceEstimation
}

META_METHODS = {
    meta.QuaNet
}

QUANTIFICATION_METHODS = AGGREGATIVE_METHODS | NON_AGGREGATIVE_METHODS | META_METHODS



