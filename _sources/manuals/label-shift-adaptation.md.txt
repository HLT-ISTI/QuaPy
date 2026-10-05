# Label Shift Adaptation

Quantification methods estimate the class prevalence values of a test sample, but a prevalence estimate is
not, by itself, a classifier that has been corrected for the shift it describes. In some scenarios you may
want the latter: a classifier whose decisions account for the (estimated) change in class priors between
the training and the test distributions, rather than a single prevalence vector.

QuaPy provides two building blocks for this:

* `qp.method.aggregative.ImportanceWeightQuantifier`, an interface implemented by the quantifiers that
  natively compute a vector of importance weights as part of estimating the target prevalence.
* `qp.classification.labelshift.LabelShiftedClassifier`, a classifier wrapper that uses any quantifier
  (whether or not it implements the interface above) to adapt a classifier to a specific, (potentially)
  shifted batch of test instances.

## The `ImportanceWeightQuantifier` interface

Some quantifiers (currently, `RLLS`, `BBSEhard`, and `BBSEsoft`; see the
{ref}`Regularized Learning under Label Shift (RLLS) <manuals/methods:Regularized Learning under Label Shift (RLLS)>`
and {ref}`Black Box Shift Estimation (BBSE) <manuals/methods:Black Box Shift Estimation (BBSE)>` sections of
the methods manual) work by first estimating a vector of importance weights

:::{math}
w_y = \frac{Q(y)}{P(y)}
:::

with :math:`P` and :math:`Q` the training and target class distributions, and only then rescaling the
training prevalence by these weights to obtain the target prevalence estimate:
:math:`\hat{Q}(y) \propto w_y \cdot P(y)`. For these methods, the weight vector is not an afterthought
computed from the prevalence estimate; it is the primitive quantity from which the estimate itself is
derived.

`ImportanceWeightQuantifier` exposes this quantity directly, in addition to the usual `predict` method
inherited from any quantifier:

* `get_importance_weights(instances)` returns the weight vector $w$ estimated for a given batch of
  (unlabelled) target instances.
* `quantify_and_weigh(instances)` returns a tuple `(prevalence, weights)`, computed from a single pass of
  classifier predictions over the instances (avoiding the double classification that calling `predict` and
  `get_importance_weights` separately would incur).

Neither method mutates any internal state of the quantifier, so both are safe to call concurrently, on the
same fitted instance, for different batches of target instances.

```python
import quapy as qp
from quapy.method.aggregative import BBSEhard
from sklearn.linear_model import LogisticRegression

train, test = qp.datasets.fetch_UCIBinaryDataset('haberman').train_test

model = BBSEhard(LogisticRegression(max_iter=2000), val_split=5)
model.fit(*train.Xy)

prevalence, weights = model.quantify_and_weigh(test.X)
```

## Adapting a classifier: `LabelShiftedClassifier`

Rescaling the classifier's posterior probabilities after the fact (as, e.g., `EMQ` does) is one way of
accounting for an estimated prevalence shift. Another common practice is instead to retrain the classifier
with a reweighted loss, via scikit-learn's `class_weight` constructor parameter; this changes the decision
function of the classifier itself, rather than only rescaling its outputs, which can matter for classifiers
whose decision boundary is not a simple function of the posterior probabilities alone.

`LabelShiftedClassifier` automates this second strategy. Given a base classifier `h` and any `quapy`
quantifier `q`, it fits `q` on the labelled training data (keeping a copy of it), and, for a given batch of
(unlabelled) target instances, estimates the importance weights: directly, via `get_importance_weights`, if
`q` is an `ImportanceWeightQuantifier`; or otherwise by dividing `q`'s estimated target prevalence by the
training prevalence. It then retrains a *fresh copy* of `h`, with `class_weight` set according to these
weights, on the original training data, and hands off classification of the target instances to this
adapted classifier. Neither the classifier instance `h` passed at construction time, nor `q`'s own internal
classifier (if any), are ever mutated.

```python
import quapy as qp
from quapy.classification.labelshift import LabelShiftedClassifier
from quapy.method.aggregative import ACC
from sklearn.linear_model import LogisticRegression

train, test = qp.datasets.fetch_UCIBinaryDataset('haberman').train_test

h = LogisticRegression(max_iter=2000)
q = ACC(LogisticRegression(max_iter=2000), val_split=5)

adapter = LabelShiftedClassifier(h, q)
adapter.fit(*train.Xy)

# label predictions and posterior probabilities from a classifier adapted to test's estimated distribution
y_pred = adapter.predict(test.X)
y_proba = adapter.predict_proba(test.X)

# or, to obtain the adapted classifier itself
adapted_h = adapter.get_classifier(test.X)
```

`h` is required to accept a `class_weight` parameter in its constructor, as `LogisticRegression` and many
other scikit-learn classifiers do; `q` can be any (fitted or unfitted) `quapy` quantifier, and need not be an
`ImportanceWeightQuantifier`.

Since the adaptation is specific to the batch of target instances given, `predict` and `predict_proba`
retrain `h` on every call. If you plan to issue several predictions against the same (suspected) shifted
distribution, call `get_classifier` once and reuse the classifier instance it returns, rather than calling
`predict`/`predict_proba` repeatedly on the same or related samples.

A small positive floor, controlled by the `weight_epsilon` constructor parameter (default `1e-4`), is
applied to the estimated weights before they are used as `class_weight`; this keeps a class with training
support from being assigned a weight of exactly `0`, which would silently drop it from the retraining loss.
