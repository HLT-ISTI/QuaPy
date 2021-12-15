from abc import ABCMeta, abstractmethod

from quapy.data import LabelledCollection


# Base Quantifier abstract class
# ------------------------------------
class BaseQuantifier(metaclass=ABCMeta):
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
        :return: `np.ndarray` of shape `(self.n_classes_,)` with class prevalence estimates.
        """
        ...

    @abstractmethod
    def set_params(self, **parameters):
        """
        Set the parameters of the quantifier.

        :param parameters: dictionary of param-value pairs
        """
        ...

    @abstractmethod
    def get_params(self, deep=True):
        """
        Return the current parameters of the quantifier.

        :param deep: for compatibility with sklearn
        :return: a dictionary of param-value pairs
        """
        ...

    @property
    @abstractmethod
    def classes_(self):
        """
        Class labels, in the same order in which class prevalence values are to be computed.

        :return: array-like
        """
        ...

    @property
    def n_classes(self):
        """
        Returns the number of classes

        :return: integer
        """
        return len(self.classes_)

    # these methods allows meta-learners to reimplement the decision based on their constituents, and not
    # based on class structure
    @property
    def binary(self):
        """
        Indicates whether the quantifier is binary or not.

        :return: False (to be overridden)
        """
        return False

    @property
    def aggregative(self):
        """
        Indicates whether the quantifier is of type aggregative or not

        :return: False (to be overridden)
        """

        return False

    @property
    def probabilistic(self):
        """
        Indicates whether the quantifier is of type probabilistic or not

        :return: False (to be overridden)
        """

        return False


class BinaryQuantifier(BaseQuantifier):
    """
    Abstract class of binary quantifiers, i.e., quantifiers estimating class prevalence values for only two classes
    (typically, to be interpreted as one class and its complement).
    """

    def _check_binary(self, data: LabelledCollection, quantifier_name):
        assert data.binary, f'{quantifier_name} works only on problems of binary classification. ' \
                            f'Use the class OneVsAll to enable {quantifier_name} work on single-label data.'

    @property
    def binary(self):
        """
        Informs that the quantifier is binary

        :return: True
        """
        return True


def isbinary(model:BaseQuantifier):
    """
    Alias for property `binary`

    :param model: the model
    :return: True if the model is binary, False otherwise
    """
    return model.binary


def isaggregative(model:BaseQuantifier):
    """
    Alias for property `aggregative`

    :param model: the model
    :return: True if the model is aggregative, False otherwise
    """

    return model.aggregative


def isprobabilistic(model:BaseQuantifier):
    """
    Alias for property `probabilistic`

    :param model: the model
    :return: True if the model is probabilistic, False otherwise
    """

    return model.probabilistic


# class OneVsAll:
#     """
#     Allows any binary quantifier to perform quantification on single-label datasets. The method maintains one binary
#     quantifier for each class, and then l1-normalizes the outputs so that the class prevelences sum up to 1.
#     """
#
#     def __init__(self, binary_method, n_jobs=-1):
#         self.binary_method = binary_method
#         self.n_jobs = n_jobs
#
#     def fit(self, data: LabelledCollection, **kwargs):
#         assert not data.binary, f'{self.__class__.__name__} expect non-binary data'
#         assert isinstance(self.binary_method, BaseQuantifier), f'{self.binary_method} does not seem to be a Quantifier'
#         self.class_method = {c: deepcopy(self.binary_method) for c in data.classes_}
#         Parallel(n_jobs=self.n_jobs, backend='threading')(
#             delayed(self._delayed_binary_fit)(c, self.class_method, data, **kwargs) for c in data.classes_
#         )
#         return self
#
#     def quantify(self, X, *args):
#         prevalences = np.asarray(
#             Parallel(n_jobs=self.n_jobs, backend='threading')(
#                 delayed(self._delayed_binary_predict)(c, self.class_method, X) for c in self.classes
#             )
#         )
#         return F.normalize_prevalence(prevalences)
#
#     @property
#     def classes(self):
#         return sorted(self.class_method.keys())
#
#     def set_params(self, **parameters):
#         self.binary_method.set_params(**parameters)
#
#     def get_params(self, deep=True):
#         return self.binary_method.get_params()
#
#     def _delayed_binary_predict(self, c, learners, X):
#         return learners[c].quantify(X)[:,1]  # the mean is the estimation for the positive class prevalence
#
#     def _delayed_binary_fit(self, c, learners, data, **kwargs):
#         bindata = LabelledCollection(data.instances, data.labels == c, n_classes=2)
#         learners[c].fit(bindata, **kwargs)


