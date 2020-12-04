from abc import ABCMeta, abstractmethod
import quapy as qp


# Base Quantifier abstract class
# ------------------------------------
class BaseQuantifier(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, data: qp.LabelledCollection, *args): ...

    @abstractmethod
    def quantify(self, instances, *args): ...

    @abstractmethod
    def set_params(self, **parameters): ...

    @abstractmethod
    def get_params(self, deep=True): ...


