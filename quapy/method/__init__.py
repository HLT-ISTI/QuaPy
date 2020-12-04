from . import aggregative as agg
from . import non_aggregative as nagg


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
    nagg.MaximumLikelihoodPrevalenceEstimation
}

QUANTIFICATION_METHODS = AGGREGATIVE_METHODS | NON_AGGREGATIVE_METHODS


# common alisases
MLPE = nagg.MaximumLikelihoodPrevalenceEstimation


