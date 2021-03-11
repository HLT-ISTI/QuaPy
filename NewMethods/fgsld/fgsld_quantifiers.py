from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from NewMethods.fgsld.fine_grained_sld import FineGrainedSLD
from quapy.method.aggregative import EMQ, CC, training_helper
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier
import quapy.functional as F


class FakeFGLSD(BaseQuantifier):
    def __init__(self, learner, nbins, isomerous, recompute_bins):
        self.learner = learner
        self.nbins = nbins
        self.isomerous = isomerous
        self.recompute_bins = recompute_bins

    def fit(self, data: LabelledCollection):
        self.Xtr, self.ytr = data.Xy
        self.learner.fit(self.Xtr, self.ytr)
        return self

    def quantify(self, instances):
        tr_priors = F.prevalence_from_labels(self.ytr, n_classes=2)
        fgsld = FineGrainedSLD(self.Xtr, instances, self.ytr, tr_priors, self.learner, n_bins=self.nbins)
        priors, posteriors = fgsld.run(self.isomerous, compute_bins_at_every_iter=self.recompute_bins)
        return priors

    def get_params(self, deep=True):
        pass

    def set_params(self, **parameters):
        pass


