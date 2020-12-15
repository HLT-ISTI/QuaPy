from . import base
from . import aggregative as agg
from . import non_aggregative


AGGREGATIVE_METHODS = {
    agg.ClassifyAndCount,
    agg.AdjustedClassifyAndCount,
    agg.ProbabilisticClassifyAndCount,
    agg.ProbabilisticAdjustedClassifyAndCount,
    agg.ExplicitLossMinimisation,
    agg.ExpectationMaximizationQuantifier,
    agg.HellingerDistanceY
}

NON_AGGREGATIVE_METHODS = {
    non_aggregative.MaximumLikelihoodPrevalenceEstimation
}

QUANTIFICATION_METHODS = AGGREGATIVE_METHODS | NON_AGGREGATIVE_METHODS



