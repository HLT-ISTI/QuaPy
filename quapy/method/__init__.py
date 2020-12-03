from . import aggregative as agg
from . import non_aggregative as nagg


AGGREGATIVE_METHODS = {
    agg.ClassifyAndCount,
    agg.AdjustedClassifyAndCount,
    agg.ProbabilisticClassifyAndCount,
    agg.ProbabilisticAdjustedClassifyAndCount,
    agg.ExplicitLossMinimisation,
    agg.ExpectationMaximizationQuantifier,
}

NON_AGGREGATIVE_METHODS = {
    nagg.MaximumLikelihoodPrevalenceEstimation
}

QUANTIFICATION_METHODS = AGGREGATIVE_METHODS | NON_AGGREGATIVE_METHODS


# common alisases
CC = agg.ClassifyAndCount
ACC = agg.AdjustedClassifyAndCount
PCC = agg.ProbabilisticClassifyAndCount
PACC = agg.ProbabilisticAdjustedClassifyAndCount
ELM = agg.ExplicitLossMinimisation
EMQ = agg.ExpectationMaximizationQuantifier
MLPE = nagg.MaximumLikelihoodPrevalenceEstimation


