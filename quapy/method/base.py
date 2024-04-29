from abc import ABCMeta, abstractmethod
from copy import deepcopy

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator

import quapy as qp
from quapy.data import LabelledCollection
import numpy as np


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


class OneVsAll:
    pass


def newOneVsAll(binary_quantifier: BaseQuantifier, n_jobs=None):
    assert isinstance(binary_quantifier, BaseQuantifier), \
        f'{binary_quantifier} does not seem to be a Quantifier'
    if isinstance(binary_quantifier, qp.method.aggregative.AggregativeQuantifier):
        return qp.method.aggregative.OneVsAllAggregative(binary_quantifier, n_jobs)
    else:
        return OneVsAllGeneric(binary_quantifier, n_jobs)


class OneVsAllGeneric(OneVsAll, BaseQuantifier):
    """
    Allows any binary quantifier to perform quantification on single-label datasets. The method maintains one binary
    quantifier for each class, and then l1-normalizes the outputs so that the class prevelence values sum up to 1.
    """

    def __init__(self, binary_quantifier: BaseQuantifier, n_jobs=None):
        assert isinstance(binary_quantifier, BaseQuantifier), \
            f'{binary_quantifier} does not seem to be a Quantifier'
        if isinstance(binary_quantifier, qp.method.aggregative.AggregativeQuantifier):
            print('[warning] the quantifier seems to be an instance of qp.method.aggregative.AggregativeQuantifier; '
                  f'you might prefer instantiating {qp.method.aggregative.OneVsAllAggregative.__name__}')
        self.binary_quantifier = binary_quantifier
        self.n_jobs = qp._get_njobs(n_jobs)

    def fit(self, data: LabelledCollection, fit_classifier=True):
        assert not data.binary, f'{self.__class__.__name__} expect non-binary data'
        assert fit_classifier == True, 'fit_classifier must be True'

        self.dict_binary_quantifiers = {c: deepcopy(self.binary_quantifier) for c in data.classes_}
        self._parallel(self._delayed_binary_fit, data)
        return self

    def _parallel(self, func, *args, **kwargs):
        return np.asarray(
            Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(func)(c, *args, **kwargs) for c in self.classes_
            )
        )

    def quantify(self, instances):
        prevalences = self._parallel(self._delayed_binary_predict, instances)
        return qp.functional.normalize_prevalence(prevalences)

    @property
    def classes_(self):
        return sorted(self.dict_binary_quantifiers.keys())

    def _delayed_binary_predict(self, c, X):
        return self.dict_binary_quantifiers[c].quantify(X)[1]

    def _delayed_binary_fit(self, c, data):
        bindata = LabelledCollection(data.instances, data.labels == c, classes=[False, True])
        self.dict_binary_quantifiers[c].fit(bindata)
