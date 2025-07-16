"""This module allows the composition of quantification methods from loss functions and feature transformations. This functionality is realized through an integration of the qunfold package: https://github.com/mirkobunse/qunfold."""

from dataclasses import dataclass
from .base import BaseQuantifier

# what to display when an ImportError is thrown
_IMPORT_ERROR_MESSAGE = """qunfold, the back-end of quapy.method.composable, is not properly installed.

To fix this error, call:

    pip install --upgrade pip setuptools wheel
    pip install "jax[cpu]"
    pip install "qunfold @ git+https://github.com/mirkobunse/qunfold@v0.1.5"
"""

# try to import members of qunfold as members of this module
try:
    import qunfold
    from qunfold.base import BaseMixin
    from qunfold.methods import AbstractMethod
    from qunfold.sklearn import CVClassifier
    from qunfold import (
        LinearMethod, # methods
        LeastSquaresLoss, # losses
        BlobelLoss,
        EnergyLoss,
        HellingerSurrogateLoss,
        CombinedLoss,
        TikhonovRegularization,
        TikhonovRegularized,
        ClassRepresentation, # representations
        HistogramRepresentation,
        DistanceRepresentation,
        KernelRepresentation,
        EnergyKernelRepresentation,
        LaplacianKernelRepresentation,
        GaussianKernelRepresentation,
        GaussianRFFKernelRepresentation,
    )
except ImportError as e:
    raise ImportError(_IMPORT_ERROR_MESSAGE) from e

__all__ = [ # control public members, e.g., for auto-documentation in sphinx
    "QUnfoldWrapper",
    "ComposableQuantifier",
    "CVClassifier",
    "LeastSquaresLoss",
    "BlobelLoss",
    "EnergyLoss",
    "HellingerSurrogateLoss",
    "CombinedLoss",
    "TikhonovRegularization",
    "TikhonovRegularized",
    "ClassRepresentation",
    "HistogramRepresentation",
    "DistanceRepresentation",
    "KernelRepresentation",
    "EnergyKernelRepresentation",
    "LaplacianKernelRepresentation",
    "GaussianKernelRepresentation",
    "GaussianRFFKernelRepresentation",
]

@dataclass
class QUnfoldWrapper(BaseQuantifier,BaseMixin):
    """A thin wrapper for using qunfold methods in QuaPy.

    Args:
      _method: An instance of `qunfold.methods.AbstractMethod` to wrap.

    Examples:
      Here, we wrap an instance of ACC to perform a grid search with QuaPy.

        >>> from qunfold import ACC
        >>> qunfold_method = QUnfoldWrapper(ACC(RandomForestClassifier(obb_score=True)))
        >>> quapy.model_selection.GridSearchQ(
        >>>     model = qunfold_method,
        >>>     param_grid = { # try both splitting criteria
        >>>         "representation__classifier__estimator__criterion": ["gini", "entropy"],
        >>>     },
        >>>     # ...
        >>> )
    """
    _method: AbstractMethod
    def fit(self, data): # data is a qp.LabelledCollection
        self._method.fit(*data.Xy, data.n_classes)
        return self
    def quantify(self, X):
        return self._method.predict(X)
    def set_params(self, **params):
        self._method.set_params(**params)
        return self
    def get_params(self, deep=True):
        return self._method.get_params(deep)
    def __str__(self):
        return self._method.__str__()

def ComposableQuantifier(loss, representation, **kwargs):
    """A generic quantification / unfolding method that solves a linear system of equations.

    This class represents any quantifier that can be described in terms of a loss function, a feature transformation, and a regularization term. In this implementation, the loss is minimized through unconstrained second-order minimization. Valid probability estimates are ensured through a soft-max trick by Bunse (2022).

    Args:
        loss: An instance of a loss class from `quapy.methods.composable`.
        representation: An instance of a representation class from `quapy.methods.composable`.
        solver (optional): The `method` argument in `scipy.optimize.minimize`. Defaults to `"trust-ncg"`.
        solver_options (optional): The `options` argument in `scipy.optimize.minimize`. Defaults to `{"gtol": 1e-8, "maxiter": 1000}`.
        seed (optional): A random number generator seed from which a numpy RandomState is created. Defaults to `None`.

    Examples:
        Here, we create the ordinal variant of ACC (Bunse et al., 2023). This variant consists of the original feature transformation of ACC and of the original loss of ACC, the latter of which is regularized towards smooth solutions.

            >>> from quapy.method.composable import (
            >>>     ComposableQuantifier,
            >>>     TikhonovRegularized,
            >>>     LeastSquaresLoss,
            >>>     ClassRepresentation,
            >>> )
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> o_acc = ComposableQuantifier(
            >>>     TikhonovRegularized(LeastSquaresLoss(), 0.01),
            >>>     ClassRepresentation(RandomForestClassifier(oob_score=True))
            >>> )

        Here, we perform hyper-parameter optimization with the ordinal ACC.

            >>> quapy.model_selection.GridSearchQ(
            >>>     model = o_acc,
            >>>     param_grid = { # try both splitting criteria
            >>>         "representation__classifier__estimator__criterion": ["gini", "entropy"],
            >>>     },
            >>>     # ...
            >>> )
        
        To use a classifier that does not provide the `oob_score` argument, such as logistic regression, you have to configure a cross validation of this classifier. Here, we employ 10 cross validation folds. 5 folds are the default.

            >>> from quapy.method.composable import CVClassifier
            >>> from sklearn.linear_model import LogisticRegression
            >>> acc_lr = ComposableQuantifier(
            >>>     LeastSquaresLoss(),
            >>>     ClassRepresentation(CVClassifier(LogisticRegression(), 10))
            >>> )
        """
    return QUnfoldWrapper(LinearMethod(loss, representation, **kwargs))
