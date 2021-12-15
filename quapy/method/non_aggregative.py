from quapy.data import LabelledCollection
from .base import BaseQuantifier


class MaximumLikelihoodPrevalenceEstimation(BaseQuantifier):
    """
    The `Maximum Likelihood Prevalence Estimation` (MLPE) method is a lazy method that assumes there is no prior
    probability shift between training and test instances (put it other way, that the i.i.d. assumpion holds).
    The estimation of class prevalence values for any test sample is always (i.e., irrespective of the test sample
    itself) the class prevalence seen during training. This method is considered to be a lower-bound quantifier that
    any quantification method should beat.
    """

    def __init__(self):
        self._classes_ = None

    def fit(self, data: LabelledCollection):
        """
        Computes the training prevalence and stores it.

        :param data: the training sample
        :return: self
        """
        self._classes_ = data.classes_
        self.estimated_prevalence = data.prevalence()
        return self

    def quantify(self, instances):
        """
        Ignores the input instances and returns, as the class prevalence estimantes, the training prevalence.

        :param instances: array-like (ignored)
        :return: the class prevalence seen during training
        """
        return self.estimated_prevalence

    @property
    def classes_(self):
        """
        Number of classes

        :return: integer
        """

        return self._classes_

    def get_params(self, deep=True):
        """
        Does nothing, since this learner has no parameters.

        :param deep: for compatibility with sklearn
        :return: `None`
        """
        return None

    def set_params(self, **parameters):
        """
        Does nothing, since this learner has no parameters.

        :param parameters: dictionary of param-value pairs (ignored)
        """
        pass
