from abc import ABCMeta, abstractmethod


# Base Quantifier abstract class
# ------------------------------------
class BaseQuantifier(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, data, *args): ...

    @abstractmethod
    def quantify(self, instances, *args): ...

    @abstractmethod
    def set_params(self, **parameters): ...

    @abstractmethod
    def get_params(self, deep=True): ...


