from typing import Union
import quapy as qp
from quapy.method.aggregative import PACC, EMQ, HDy
import quapy.functional as F


class PACCSLD(PACC):
    """
    This method combines the EMQ improved posterior probabilities with PACC.
    Note: the posterior probabilities are re-calibrated with EMQ only during prediction, and not also during fit since,
    for PACC, the validation split is known to have the same prevalence as the training set (this is because the split
    is stratified) and thus the posterior probabilities should not be re-calibrated for a different prior (it actually
    happens to degrades performance).
    """

    def fit(self, data: qp.data.LabelledCollection, fit_learner=True, val_split:Union[float, int, qp.data.LabelledCollection]=0.4):
        self.train_prevalence = F.prevalence_from_labels(data.labels, data.n_classes)
        return super(PACCSLD, self).fit(data, fit_learner, val_split)

    def aggregate(self, classif_posteriors):
        priors, posteriors = EMQ.EM(self.train_prevalence, classif_posteriors, epsilon=1e-4)
        return super(PACCSLD, self).aggregate(posteriors)


class HDySLD(HDy):
    """
        This method combines the EMQ improved posterior probabilities with HDy.
        Note: [same as PACCSLD]
        """
    def fit(self, data: qp.data.LabelledCollection, fit_learner=True,
            val_split: Union[float, int, qp.data.LabelledCollection] = 0.4):
        self.train_prevalence = F.prevalence_from_labels(data.labels, data.n_classes)
        return super(HDySLD, self).fit(data, fit_learner, val_split)

    def aggregate(self, classif_posteriors):
        priors, posteriors = EMQ.EM(self.train_prevalence, classif_posteriors, epsilon=1e-4)
        return super(HDySLD, self).aggregate(posteriors)
