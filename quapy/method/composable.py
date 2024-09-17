"""This module allows the composition of quantification methods from loss functions and feature transformations. This functionality is realized through an integration of the qunfold package: https://github.com/mirkobunse/qunfold."""

_import_error_message = """qunfold, the back-end of quapy.method.composable, is not properly installed.

To fix this error, call:

    pip install --upgrade pip setuptools wheel
    pip install "jax[cpu]"
    pip install "qunfold @ git+https://github.com/mirkobunse/qunfold@v0.1.4"
"""

try:
    import qunfold
    from qunfold.quapy import QuaPyWrapper
    from qunfold.sklearn import CVClassifier
    from qunfold import (
        LeastSquaresLoss, # losses
        BlobelLoss,
        EnergyLoss,
        HellingerSurrogateLoss,
        CombinedLoss,
        TikhonovRegularization,
        TikhonovRegularized,
        ClassTransformer, # transformers
        HistogramTransformer,
        DistanceTransformer,
        KernelTransformer,
        EnergyKernelTransformer,
        LaplacianKernelTransformer,
        GaussianKernelTransformer,
        GaussianRFFKernelTransformer,
    )

    __all__ = [ # control public members, e.g., for auto-documentation in sphinx; omit QuaPyWrapper
        "ComposableQuantifier",
        "CVClassifier",
        "LeastSquaresLoss",
        "BlobelLoss",
        "EnergyLoss",
        "HellingerSurrogateLoss",
        "CombinedLoss",
        "TikhonovRegularization",
        "TikhonovRegularized",
        "ClassTransformer",
        "HistogramTransformer",
        "DistanceTransformer",
        "KernelTransformer",
        "EnergyKernelTransformer",
        "LaplacianKernelTransformer",
        "GaussianKernelTransformer",
        "GaussianRFFKernelTransformer",
    ]
except ImportError as e:
    raise ImportError(_import_error_message) from e

def ComposableQuantifier(loss, transformer, **kwargs):
    """A generic quantification / unfolding method that solves a linear system of equations.

    This class represents any quantifier that can be described in terms of a loss function, a feature transformation, and a regularization term. In this implementation, the loss is minimized through unconstrained second-order minimization. Valid probability estimates are ensured through a soft-max trick by Bunse (2022).

    Args:
        loss: An instance of a loss class from `quapy.methods.composable`.
        transformer: An instance of a transformer class from `quapy.methods.composable`.
        solver (optional): The `method` argument in `scipy.optimize.minimize`. Defaults to `"trust-ncg"`.
        solver_options (optional): The `options` argument in `scipy.optimize.minimize`. Defaults to `{"gtol": 1e-8, "maxiter": 1000}`.
        seed (optional): A random number generator seed from which a numpy RandomState is created. Defaults to `None`.

    Examples:
        Here, we create the ordinal variant of ACC (Bunse et al., 2023). This variant consists of the original feature transformation of ACC and of the original loss of ACC, the latter of which is regularized towards smooth solutions.

            >>> from quapy.method.composable import (
            >>>     ComposableQuantifier,
            >>>     TikhonovRegularized,
            >>>     LeastSquaresLoss,
            >>>     ClassTransformer,
            >>> )
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> o_acc = ComposableQuantifier(
            >>>     TikhonovRegularized(LeastSquaresLoss(), 0.01),
            >>>     ClassTransformer(RandomForestClassifier(oob_score=True))
            >>> )

        Here, we perform hyper-parameter optimization with the ordinal ACC.

            >>> quapy.model_selection.GridSearchQ(
            >>>     model = o_acc,
            >>>     param_grid = { # try both splitting criteria
            >>>         "transformer__classifier__estimator__criterion": ["gini", "entropy"],
            >>>     },
            >>>     # ...
            >>> )
        
        To use a classifier that does not provide the `oob_score` argument, such as logistic regression, you have to configure a cross validation of this classifier. Here, we employ 10 cross validation folds. 5 folds are the default.

            >>> from quapy.method.composable import CVClassifier
            >>> from sklearn.linear_model import LogisticRegression
            >>> acc_lr = ComposableQuantifier(
            >>>     LeastSquaresLoss(),
            >>>     ClassTransformer(CVClassifier(LogisticRegression(), 10))
            >>> )
        """
    return QuaPyWrapper(qunfold.GenericMethod(loss, transformer, **kwargs))
