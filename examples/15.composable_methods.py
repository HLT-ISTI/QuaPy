"""
This example illustrates the composition of quantification methods from
arbitrary loss functions and feature representations. It will extend the basic
example on the usage of quapy with this composition.

This example requires the installation of qunfold, the back-end of QuaPy's
composition module:

    pip install --upgrade pip setuptools wheel
    pip install "jax[cpu]"
    pip install "qunfold @ git+https://github.com/mirkobunse/qunfold@v0.1.5"
"""

import numpy as np
import quapy as qp
import quapy.functional as F

# First of all, we load the same data as in the basic example.

data = qp.data.preprocessing.text2tfidf(
    qp.datasets.fetch_reviews("hp"),
    min_df = 5,
)
training, testing = data.train_test

# We start by recovering PACC from its building blocks, a LeastSquaresLoss and
# a probabilistic ClassRepresentation. A 5-fold cross-validation is implemented
# through a CVClassifier.

from quapy.method.composable import (
    ComposableQuantifier,
    LeastSquaresLoss,
    ClassRepresentation,
    CVClassifier,
)
from sklearn.linear_model import LogisticRegression

pacc = ComposableQuantifier(
    LeastSquaresLoss(),
    ClassRepresentation(
        CVClassifier(LogisticRegression(random_state=0), 5),
        is_probabilistic = True
    ),
)

# Let's evaluate this quantifier.

print(f"Evaluating PACC: {pacc}")
pacc.fit(training)
app = qp.protocol.APP(testing, sample_size=100, n_prevalences=21, repeats=1)
absolute_errors = qp.evaluation.evaluate(
    model = pacc,
    protocol = app,
    error_metric = "ae",
)
print(f"MAE = {np.mean(absolute_errors):.4f}+-{np.std(absolute_errors):.4f}")

# We now turn to the composition of novel methods. As an example, we use the
# (squared) Hellinger distance as a loss function but, unlike HDy, we do not
# compute any histograms from the output of the classifier.

from quapy.method.composable import HellingerSurrogateLoss

model = ComposableQuantifier(
    HellingerSurrogateLoss(), # the loss is different from before
    ClassRepresentation( # we use the same representation
        CVClassifier(LogisticRegression(random_state=0), 5),
        is_probabilistic = True
    ),
)

print(f"Evaluating {model}")
model.fit(training)
absolute_errors = qp.evaluation.evaluate(
    model = model,
    protocol = app, # use the same protocol for evaluation
    error_metric = "ae",
)
print(f"MAE = {np.mean(absolute_errors):.4f}+-{np.std(absolute_errors):.4f}")

# In general, any composed method solves a linear system of equations by
# minimizing the loss after representing the data. Methods of this kind include
# ACC, PACC, HDx, HDy, and many other well-known methods, as well as an
# unlimited number of re-combinations of their building blocks.

# To illustrate hyper-parameter optimization, we now define a method that
# employs a weighted sum of the LeastSquaresLoss and the
# HellingerSurrogateLoss. We will consider both the weighting of these losses
# and the C parameter of the LogisticRegression as hyper-parameters to be
# optimized.

from quapy.method.composable import CombinedLoss

model = ComposableQuantifier(
    CombinedLoss(HellingerSurrogateLoss(), LeastSquaresLoss()),
    ClassRepresentation(
        CVClassifier(LogisticRegression(random_state=0), 5),
        is_probabilistic = True
    ),
)

from quapy.method.composable import QUnfoldWrapper
from qunfold import LinearMethod

model = QUnfoldWrapper(LinearMethod(
    CombinedLoss(HellingerSurrogateLoss(), LeastSquaresLoss()),
    ClassRepresentation(
        CVClassifier(LogisticRegression(random_state=0), 5),
        is_probabilistic = True
    ),
))

# The names of the parameters stem from the comparably deep object hierarchy
# that composable methods define.

param_grid = {
    "loss__weights": [ (w, 1-w) for w in [.1, .5, .9] ],
    "representation__classifier__estimator__C": [1e-1, 1e1],
}

grid_search = qp.model_selection.GridSearchQ(
    model = model,
    param_grid = param_grid,
    protocol = app, # use the protocol that we used for testing before
    error = "mae",
    refit = False,
    verbose = True,
).fit(training)
print(
    f"Best hyper-parameters = {grid_search.best_params_}",
    f"Best MAE = {grid_search.best_score_}",
    sep = "\n",
)

# Note that a proper evaluation would still require the best model to be
# evaluated on a separate test set.

# To implement your own loss functions and feature representations, please
# follow the corresponding manual of the qunfold package. This package provides
# the back-end of QuaPy’s composable module and is fully compatible with QuaPy.
#
# https://mirkobunse.github.io/qunfold/developer-guide.html#custom-implementations
