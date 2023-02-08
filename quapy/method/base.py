from abc import ABCMeta, abstractmethod
from copy import deepcopy

from sklearn.base import BaseEstimator

import quapy as qp
from quapy.data import LabelledCollection


# Base Quantifier abstract class
# ------------------------------------
class BaseQuantifier(BaseEstimator):
    """
    Abstract Quantifier. A quantifier is defined as an object of a class that implements the method :meth:`fit` on
    :class:`quapy.data.base.LabelledCollection`, the method :meth:`quantify`, and the :meth:`set_params` and
    :meth:`get_params` for model selection (see :meth:`quapy.model_selection.GridSearchQ`)
    """

    @abstractmethod
    def fit(self, data: LabelledCollection):
        """
        Trains a quantifier.

        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
        :return: self
        """
        ...

    @abstractmethod
    def quantify(self, instances):
        """
        Generate class prevalence estimates for the sample's instances

        :param instances: array-like
        :return: `np.ndarray` of shape `(n_classes,)` with class prevalence estimates.
        """
        ...


class BinaryQuantifier(BaseQuantifier):
    """
    Abstract class of binary quantifiers, i.e., quantifiers estimating class prevalence values for only two classes
    (typically, to be interpreted as one class and its complement).
    """

    def _check_binary(self, data: LabelledCollection, quantifier_name):
        assert data.binary, f'{quantifier_name} works only on problems of binary classification. ' \
                            f'Use the class OneVsAll to enable {quantifier_name} work on single-label data.'


class OneVsAllGeneric:
    """
    Allows any binary quantifier to perform quantification on single-label datasets. The method maintains one binary
    quantifier for each class, and then l1-normalizes the outputs so that the class prevelence values sum up to 1.
    """

    def __init__(self, binary_quantifier, n_jobs=None):
        assert isinstance(binary_quantifier, BaseQuantifier), \
            f'{binary_quantifier} does not seem to be a Quantifier'
        self.binary_quantifier = binary_quantifier
        self.n_jobs = qp._get_njobs(n_jobs)

    def fit(self, data: LabelledCollection, **kwargs):
        assert not data.binary, \
            f'{self.__class__.__name__} expect non-binary data'
        self.class_quatifier = {c: deepcopy(self.binary_quantifier) for c in data.classes_}
        Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(self._delayed_binary_fit)(c, self.class_quatifier, data, **kwargs) for c in data.classes_
        )
        return self

    def quantify(self, X, *args):
        prevalences = np.asarray(
            Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(self._delayed_binary_predict)(c, self.class_quatifier, X) for c in self.classes
            )
        )
        return F.normalize_prevalence(prevalences)

    @property
    def classes(self):
        return sorted(self.class_quatifier.keys())

    def set_params(self, **parameters):
        self.binary_quantifier.set_params(**parameters)

    def get_params(self, deep=True):
        return self.binary_quantifier.get_params()

    def _delayed_binary_predict(self, c, quantifiers, X):
        return quantifiers[c].quantify(X)[:, 1]  # the mean is the estimation for the positive class prevalence

    def _delayed_binary_fit(self, c, quantifiers, data, **kwargs):
        bindata = LabelledCollection(data.instances, data.labels == c, n_classes=2)
        quantifiers[c].fit(bindata, **kwargs)


