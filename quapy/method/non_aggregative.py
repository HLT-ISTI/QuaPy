from quapy.data import LabelledCollection
from .base import BaseQuantifier



class MaximumLikelihoodPrevalenceEstimation(BaseQuantifier):

    def __init__(self, **kwargs):
        pass

    def fit(self, data: LabelledCollection, *args):
        self.estimated_prevalence = data.prevalence()

    def quantify(self, documents, *args):
        return self.estimated_prevalence

    def get_params(self):
        pass

    def set_params(self, **parameters):
        pass
