from quapy.data import LabelledCollection
from .base import BaseQuantifier


class MaximumLikelihoodPrevalenceEstimation(BaseQuantifier):

    def __init__(self, **kwargs):
        self._classes_ = None

    def fit(self, data: LabelledCollection, *args):
        self._classes_ = data.classes_
        self.estimated_prevalence = data.prevalence()

    def quantify(self, documents, *args):
        return self.estimated_prevalence

    @property
    def classes_(self):
        return self._classes_

    def get_params(self):
        pass

    def set_params(self, **parameters):
        pass
