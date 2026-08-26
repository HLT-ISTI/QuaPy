# Quantification Methods

Quantification methods can be categorized as belonging to
`aggregative`, `non-aggregative`, and `meta-learning` groups. 
`Aggregative` quantifiers rely on a surrogate classifier as an intermediate
step, and devise different aggregation functions over the classifier outputs.
By contrast, `non-aggregative` methods perform quantification without requiring
an underlying classifier.
`Meta-learning` refers to quantification methods that are constructed over simpler
quantification methods, and implement high-level orchestration functions.

Beyond these three traditional categories of methods, we here present an additional,
orthogonal one: `Bayesian quantifiers`, i.e., quantification methods that do not simply
return point-estimates of class prevalence, but are also able to provide a measure of
uncertaintly around them. 

Any quantifier in QuaPy shoud extend the class `BaseQuantifier`,
and implement some abstract methods:
```python
    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...
```
The meaning of those functions should be familiar to those
used to work with scikit-learn since the class structure of QuaPy
is directly inspired by scikit-learn's _Estimators_. Functions
`fit` and `predict` (for which there is an alias `quantify`) 
are used to train the model and to provide
class estimations.
Quantifiers also extend from scikit-learn's `BaseEstimator`, in order
to simplify the use of `set_params` and `get_params` used in 
[model selection](./model-selection).

## Aggregative Methods

All quantification methods are implemented as part of the
`qp.method` package. In particular, `aggregative` methods are defined in
`qp.method.aggregative`, and extend `AggregativeQuantifier(BaseQuantifier)`.
The methods that any `aggregative` quantifier must implement are:

```python
    @abstractmethod
    def aggregation_fit(self, classif_predictions, labels):

    @abstractmethod
    def aggregate(self, classif_predictions): ...
```

The argument `classif_predictions` is whatever the method `classify` returns. 
QuaPy comes with default implementations that cover most common cases, but you can
override `classify` in case your method requires further or different information to work.

These two functions replace the `fit` and `predict` methods, which
come with default implementations. For instance, the `fit` function is 
provided and amounts to: 

```python
    def fit(self, X, y):
        self._check_init_parameters()
        classif_predictions, labels = self.classifier_fit_predict(X, y)
        self.aggregation_fit(classif_predictions, labels)
        return self
```

Note that this function fits the classifier, and generates the predictions. This is assumed
to be a routine common to all aggregative quantifiers, and is provided by QuaPy. What remains 
ahead is to define the `aggregation_fit` function, that takes as input the classifier predictions 
and the original training data (this latter is typically unused). The classifier predictions
can be:
- confidence scores: quantifiers inheriting directly from `AggregativeQuantifier`
- crisp predictions: quantifiers inheriting from `AggregativeCrispQuantifier`
- posterior probabilities: quantifiers inheriting from `AggregativeSoftQuantifier`
- _anything_: custom quantifiers overriding the `classify` method

Note also that the `fit` method also calls `_check_init_parameters`; this function is meant to be
overriden (if needed) and allows the method to quickly raise any exception based on any inconsistency
found in the `__init__` arguments, thus avoiding to break after training the classifier and generating
predictions.

Similarly, the function `predict` (alias `quantify`) is provided, and amounts to:

```python
def predict(self, X):
    classif_predictions = self.classify(X)
    return self.aggregate(classif_predictions)
```

in which only the function `aggregate` is required to be overriden in most cases.

Aggregative quantifiers are expected to maintain a classifier (which is
accessed through the `@property` `classifier`). This classifier is
given as input to the quantifier, and will be trained by the quantifier's fit (default).
Alternatively, the classifier can be already fit on external data; in this case, the `fit_learner` 
argument in the `__init__` should be set to False (see [4.using_pretrained_classifier.py](https://github.com/HLT-ISTI/QuaPy/blob/master/examples/4.using_pretrained_classifier.py)
for a full code example).

The above patterns (in training: (i) fit the classifier, then (ii) fit the aggregation; 
in test: (i) classify, then (ii) aggregate) allows QuaPy to optimize many internal procedures, 
on the grounds that steps (i) are slower than steps (ii). 
In particular, the model selection routing takes advantage of this two-step process
and generates classifiers only for the valid combinations of hyperparameters of the 
classifier, and then _clones_ these classifiers and explores the combinations
of hyperparameters that are specific to the quantifier (this can result in huge
time savings).
Concerning the inference phase, this two-step process allow the evaluation of many 
standard protocols (e.g., the [artificial sampling protocol](./evaluation)) to be
carried out very efficiently. The reason is that the entire set can be pre-classified
once, and the quantification estimations for different samples can directly
reuse these predictions, without requiring to classify each element every time.
QuaPy leverages this property to speed-up any procedure having to do with
quantification over samples, as is customarily done in model selection or 
in evaluation.

### The Classify & Count variants

QuaPy implements the four CC variants, i.e.:

* _CC_ (Classify & Count), the simplest aggregative quantifier; one that
  classifies all instances and computes the prevalence of the predicted labels.
  This baseline is discussed, among others, in [Forman (2008)](https://link.springer.com/article/10.1007/s10618-008-0097-y).
* _ACC_ (Adjusted Classify & Count), the adjusted variant of CC, originally
  proposed in [Forman (2008)](https://link.springer.com/article/10.1007/s10618-008-0097-y).
* _PCC_ (Probabilistic Classify & Count), the probabilistic variant of CC that
  relies on the posterior probabilities returned by a probabilistic classifier,
  introduced in [Bella et al. (2010)](https://ieeexplore.ieee.org/abstract/document/5694031).
* _PACC_ (Probabilistic Adjusted Classify & Count), the adjusted variant of PCC,
  also introduced in [Bella et al. (2010)](https://ieeexplore.ieee.org/abstract/document/5694031).

The following code serves as a complete example using CC equipped 
with a SVM as the classifier:

```python
import quapy as qp
import quapy.functional as F
from sklearn.svm import LinearSVC

training, test = qp.datasets.fetch_twitter('hcr', pickle=True).train_test
Xtr, ytr = training.Xy

# instantiate a classifier learner, in this case a SVM
svm = LinearSVC()

# instantiate a Classify & Count with the SVM
# (an alias is available in qp.method.aggregative.ClassifyAndCount)
model = qp.method.aggregative.CC(svm)
model.fit(Xtr, ytr)
estim_prevalence = model.predict(test.X)
```

The same code could be used to instantiate an ACC, by simply replacing
the instantiation of the model with:
```python
model = qp.method.aggregative.ACC(svm)
```
Note that the adjusted variants (ACC and PACC) need to estimate
some parameters for performing the adjustment (e.g., the 
_true positive rate_ and the _false positive rate_ in case of
binary classification) that are estimated on a validation split
of the labelled set. In this case, the `__init__` method of
ACC defines an additional parameter, `val_split`. If this parameter
is set to a float in [0,1] representing a fraction (e.g., 0.4) 
then that fraction of labelled data (e.g., 40%) 
will be used for estimating the parameters for adjusting the
predictions. This parameters can also be set with an integer,
indicating that the parameters should be estimated by means of
_k_-fold cross-validation, for which the integer indicates the
number _k_ of folds (the default value is 5). Finally, `val_split` can be set to a 
specific held-out validation set (i.e., an tuple `(X,y)`).

The following code illustrates the case in which PCC is used:

```python
model = qp.method.aggregative.PCC(svm)
model.fit(Xtr, ytr)
estim_prevalence = model.predict(Xte)
print('classifier:', model.classifier)
```
In this case, QuaPy will print:
```
The learner LinearSVC does not seem to be probabilistic. The learner will be calibrated.
classifier: CalibratedClassifierCV(base_estimator=LinearSVC(), cv=5)
```
The first output indicates that the learner (`LinearSVC` in this case)
is not a probabilistic classifier (i.e., it does not implement the 
`predict_proba` method) and so, the classifier will be converted to
a probabilistic one through [calibration](https://scikit-learn.org/stable/modules/calibration.html).
As a result, the classifier that is printed in the second line points
to a `CalibratedClassifierCV` instance. Note that calibration can only
be applied to hard classifiers if `fit_learner=True`; an exception 
will be raised otherwise.

Lastly, everything we said about ACC and PCC
applies to PACC as well.

A Bayesian counterpart of the ACC family is also available; see the
{ref}`Bayesian Quantification Methods section <manuals/methods:Bayesian Quantification Methods>`
for `BayesianCC`.

_New in v0.1.9_: quantifiers ACC and PACC now have three additional arguments: `method`, `solver` and `norm`:

* Argument `method` specifies how to solve, for `p`, the linear system `q = Mp` (where `q` is the unadjusted counts for the
test sample, `M` contains the class-conditional unadjusted counts --i.e., the missclassification rates-- and `p` is the
sought prevalence vector):
    * option `"inversion"`: attempts to invert matrix `M`, thus solving `Minv q = p`. In degenerated cases, this
      inversion may not exist. In such cases, the method defaults to returning `q` (the unadjusted counts)
    * option `"invariant-ratio""` uses the invariant ratio estimator system proposed in Remark 5 of 
[Vaz, A.F., Izbicki F. and Stern, R.B. "Quantification Under Prior Probability Shift: the Ratio Estimator 
and its Extensions", in Journal of Machine Learning Research 20 (2019)](https://jmlr.csail.mit.edu/papers/volume20/18-456/18-456.pdf).

* Argument `solver` specifies how to solve the linear system.
  * `"exact-raise"` solves the system of linear equations and raises an exception if the system is not solvable
  * `"exact-cc"` returns the original unadjusted count if the system is not solvable 
  * `"minimize"`  minimizes the L2 norm of :math:`|Mp-q|`. This one generally works better, and is the
          default parameter. More details about this can be consulted in 
  [Bunse, M. "On Multi-Class Extensions of Adjusted Classify and Count", 
  on proceedings of the 2nd International Workshop on Learning to Quantify: Methods and Applications (LQ 2022), 
  ECML/PKDD 2022, Grenoble (France)](https://lq-2022.github.io/proceedings/CompleteVolume.pdf)).

* Argument `norm` specifies how to normalize the estimate `p` when the vector lies outside of the probability simplex. 
Options are:
  * `"clip"` which clips the values to range `[0, 1]` and then L1-normalizes the vector
  * `"mapsimplex"` which projects the results on the probability simplex, as proposed by Vaz et al. in 
  [Remark 5 of Vaz, et. (2019)](https://jmlr.csail.mit.edu/papers/volume20/18-456/18-456.pdf). This implementation 
  relies on [Mathieu Blondel's `projection_simplex_sort`](https://gist.github.com/mblondel/6f3b7aaad90606b98f71))
  * `"condsoftmax"`  applies softmax normalization only if the prevalence vector lies outside of the probability simplex.


### Threshold Optimization methods

QuaPy implements Forman's threshold optimization methods;
see, e.g., [(Forman 2006)](https://dl.acm.org/doi/abs/10.1145/1150402.1150423) 
and [(Forman 2008)](https://link.springer.com/article/10.1007/s10618-008-0097-y).
These include: `T50`, `MAX`, `X`, Median Sweep (`MS`), and its variant `MS2`.

These methods are binary-only and implement different heuristics for 
improving the stability of the denominator of the ACC adjustment (`tpr-fpr`).
The methods are called "threshold" since said heuristics have to do
with different choices of the underlying classifier's threshold.


### Expectation Maximization (EMQ) / Maximum Likelihood for Label Shift (MLLS)

The Expectation Maximization Quantifier (EMQ) (also known as
SLD after the name of the proponets, or Maximum Likelihood for Label Shift, MLLS) , is available at `qp.method.aggregative.EMQ` or via the 
alias `qp.method.aggregative.ExpectationMaximizationQuantifier`. 
The method is described in:

_Saerens, M., Latinne, P., and Decaestecker, C. (2002). Adjusting the outputs of a classifier
to new a priori probabilities: A simple procedure. Neural Computation, 14(1):21–41._

EMQ works with a probabilistic classifier (if the classifier
given as input is a hard one, a calibration will be attempted).
Although this method was originally proposed for improving the
posterior probabilities of a probabilistic classifier, and not
for improving the estimation of prior probabilities, EMQ ranks 
almost always among the most effective quantifiers in the
experiments we have carried out.

An example of use can be found below:

```python
import quapy as qp
from sklearn.linear_model import LogisticRegression

train, test = qp.datasets.fetch_twitter('hcr', pickle=True).train_test

model = qp.method.aggregative.EMQ(LogisticRegression())
model.fit(*train.Xy)
estim_prevalence = model.predict(test.X)
```

EMQ accepts additional parameters in the construction method:
* `exact_train_prev`: set to True for using the true training prevalence as the departing
prevalence estimation (default behaviour), or to False for using an approximation of it as
suggested by [Alexandari et al. (2020)](http://proceedings.mlr.press/v119/alexandari20a.html) 
* `calib`: allows to indicate a calibration method, among those
proposed by [Alexandari et al. (2020)](http://proceedings.mlr.press/v119/alexandari20a.html),
including the Bias-Corrected Temperature Scaling 
(`bcts`), Vector Scaling (`bcts`), No-Bias Temperature Scaling (`nbvs`), 
or Temperature Scaling (`ts`); default is `None` (no calibration).
* `on_calib_error`: indicates the policy to follow in case the calibrator fails at runtime.
        Options include `raise` (default), in which case a RuntimeException is raised; and `backup`, in which
        case the calibrator is silently skipped.

You can use the class method `EMQ_BCTS` to effortlessly instantiate EMQ with the best performing
heuristics found by [Alexandari et al. (2020)](http://proceedings.mlr.press/v119/alexandari20a.html). See the API documentation for further details. 

For a Bayesian label-shift counterpart based on the same general family of ideas,
see the {ref}`Bayesian Quantification Methods section <manuals/methods:Bayesian Quantification Methods>`
for `BayesianMAPLS`.

### Regularized Learning under Label Shift (RLLS)

`RLLS` is available at `qp.method.aggregative.RLLS` and ports the regularized
importance-weight estimation procedure of
[Azizzadenesheli, K., Liu, A., Yang, F., and Anandkumar, A. (2019). Regularized
Learning for Domain Adaptation under Label Shifts.
ICLR 2019](https://arxiv.org/abs/1903.09734) to QuaPy's aggregative interface.
The method estimates the label-shift importance weights `w = q(y)/p(y)` from
the classifier's validation posteriors (or, in `mode='hard'`, its argmax
predictions) and the corresponding source labels, regularizing the estimation
by an amount controlled by `alpha` (scaled by a finite-sample confidence term
governed by `delta`). The resulting weights are then used to rescale the
training prevalence into the target prevalence estimate.

Like ACC and PACC, RLLS requires validation predictions and therefore expects
`val_split` to be set (as an integer for k-fold cross-validation, a float for
a held-out split, or an explicit `(X, y)` tuple) whenever `fit_classifier=True`.
This method relies on the optional `cvxpy` dependency, which must be
installed separately (`$ pip install cvxpy`).

```python
import quapy as qp
from quapy.method.aggregative import RLLS
from sklearn.linear_model import LogisticRegression

train, test = qp.datasets.fetch_UCIBinaryDataset('haberman').train_test

model = RLLS(LogisticRegression(max_iter=2000), val_split=5)
model.fit(*train.Xy)
estim_prevalence = model.predict(test.X)
```

RLLS computes its importance weights directly (it implements the `ImportanceWeightQuantifier` interface); see
the {ref}`Label Shift Adaptation manual <manuals/label-shift-adaptation:Label Shift Adaptation>` for how to
access these weights, or use them to adapt a classifier itself rather than only estimating prevalence.

### Black Box Shift Estimation (BBSE)

`BBSEhard` and `BBSEsoft` are available at `qp.method.aggregative.BBSEhard` and
`qp.method.aggregative.BBSEsoft`, respectively, and implement the Black Box Shift Estimator
proposed in:

_Lipton, Z., Wang, Y. X., & Smola, A. (2018, July). Detecting and correcting for label shift
with black box predictors. In International conference on machine learning
(pp. 3122-3130). PMLR._ ([link to paper](https://proceedings.mlr.press/v80/lipton18a.html))

BBSE is similar in spirit to ACC and PACC in that it exploits the label-shift invariance of
`P(hat{Y}|Y)` to correct for the change in class prevalence between the training and the test
distributions. However, while ACC solves the linear system `q = Mp` (with `M` the matrix of
class-conditional misclassification rates and `p` the sought prevalence vector), BBSE instead
solves `q = Cw` for the importance-weight vector `w`, with `w_i = Q(i)/P(i)` the ratio between
the target and the source class priors, and `C` the joint-distribution matrix
`C_ij = P(hat{Y}=i, Y=j)` estimated on a validation split. The target prevalence estimate is
then recovered as `Q(y) = w_y * P(y)`.

`BBSEhard` estimates `C` from crisp classifier predictions (i.e., a standard confusion matrix,
normalized to sum to 1), while `BBSEsoft` estimates it from the classifier's posterior
probabilities instead, in the same spirit in which PACC generalizes ACC.

```python
import quapy as qp
from quapy.method.aggregative import BBSEhard, BBSEsoft
from sklearn.linear_model import LogisticRegression

train, test = qp.datasets.fetch_UCIBinaryDataset('haberman').train_test

model = BBSEhard(LogisticRegression(max_iter=2000), val_split=5)
model.fit(*train.Xy)
estim_prevalence = model.predict(test.X)

# or, using posterior probabilities instead of crisp counts:
model = BBSEsoft(LogisticRegression(max_iter=2000), val_split=5)
model.fit(*train.Xy)
estim_prevalence = model.predict(test.X)
```

As with ACC and RLLS, both variants require validation predictions and therefore expect
`val_split` to be set whenever `fit_classifier=True`. They also accept the same `solver`
(`"minimize"`, `"exact-raise"`, `"exact-cc"`) and `norm` (`"clip"`, `"mapsimplex"`,
`"condsoftmax"`) arguments discussed above for ACC/PACC.

Like RLLS, both `BBSEhard` and `BBSEsoft` implement the `ImportanceWeightQuantifier` interface; see the
{ref}`Label Shift Adaptation manual <manuals/label-shift-adaptation:Label Shift Adaptation>` for how to
access these weights directly, or use them to adapt a classifier itself rather than only estimating
prevalence.

### Distribution Matching 

Distribution Matching (DM) methods search for the mixture parameter (the sought class prevalence values)
yielding the mixture between the class-wise representations that best matches the test distribution.
Different criteria for deciding how this matching is assessed, and different ways for modelling the
distributions give rise to different instantiations of DM methods.

The following methods are here discussed because they rely on a surrogate classifier for representing
the distributions, albeit different non-aggregative variants of them do often exist. Aside from this,
the formulation of DM methods is flexible enough as to accomodate methods that were proposed under a different
framework; examples include ACC and PACC. 

See the frameworks by [Firat](https://arxiv.org/abs/1606.00868), [Bunse](https://dl.gi.de/items/5a61f30f-6c84-4165-bd92-9098bd9e91aa), [Garg et al.](https://dl.acm.org/doi/10.5555/3495724.3496001), or [Dussap](https://theses.hal.science/tel-04931123), for more details.

#### Hellinger Distance y (HDy)

Implementation of the method based on the Hellinger Distance y (HDy) proposed by
[González-Castro, V., Alaiz-Rodríguez, R., and Alegre, E. (2013). Class distribution
estimation based on the Hellinger distance. Information Sciences, 218:146-164.](https://www.sciencedirect.com/science/article/pii/S0020025512004069)

It is implemented in `qp.method.aggregative.HDy` (also accessible
through the allias `qp.method.aggregative.HellingerDistanceY`).
This method works with a probabilistic classifier (hard classifiers
can be used as well and will be calibrated) and requires a validation
set to estimate parameter for the mixture model. Just like 
ACC and PACC, this quantifier receives a `val_split` argument
in the constructor that can either be a float indicating the proportion
of training data to be taken as the validation set (in a random
stratified split), or the validation set itself (i.e., an tuple
`(X,y)`). 

HDy was proposed as a binary classifier and the implementation
provided in QuaPy accepts only binary datasets. 
 
The following code shows an example of use:

```python
import quapy as qp
from sklearn.linear_model import LogisticRegression

# load a binary dataset
dataset = qp.datasets.fetch_reviews('hp', pickle=True)
qp.data.preprocessing.text2tfidf(dataset, min_df=5, inplace=True)

model = qp.method.aggregative.HDy(LogisticRegression())
model.fit(*dataset.training.Xy)
estim_prevalence = model.predict(dataset.test.X)
```

#### Generalized Distribution Matching y (DMy)

QuaPy also provides a generalized posterior-space distribution-matching
quantifier for binary or multiclass problems, implemented as
`qp.method.aggregative.DMy`. This class follows the generic distribution
matching view discussed by [Firat (2016)](https://arxiv.org/abs/1606.00868):
it represents class-conditional posterior distributions by histograms and then
searches for the prevalence vector whose mixture best matches the test
distribution.

`DMy` is intentionally flexible and exposes three main design choices: the
number of histogram bins (`nbins`), the divergence to minimize (`divergence`,
e.g., `'HD'` or `'topsoe'`), and whether to match PDFs or CDFs (`cdf`). The
optimization routine can also be selected through `search`; the default
`'optim_minimize'` works for multiclass problems, while `'linear_search'` and
`'ternary_search'` are binary-only. A multiclass HDy-like instance can be
obtained as:

```python
multiclass_hdy = qp.method.aggregative.DMy(
    classifier=LogisticRegression(),
    divergence='HD',
    cdf=False,
)
```

#### DyS

QuaPy implements the binary `DyS` framework proposed by
[Maletzke et al. (2020)](https://ojs.aaai.org/index.php/AAAI/article/view/4376)
as `qp.method.aggregative.DyS`. Conceptually, `DyS` can be seen as a
generalization of HDy in which the prevalence is found by ternary search over a
distribution-matching objective. In QuaPy, the user can select the number of
histogram bins (`n_bins`), the divergence (`divergence`), and the optimization
tolerance (`tol`).

#### Energy Distance y (EDy)

QuaPy also adapts `EDy` from [quantificationlib](https://github.com/AICGijon/quantificationlib), 
which is available as `qp.method.aggregative.EDy`. 

This
method replaces histogram matching with an energy-distance formulation defined
directly on posterior-probability vectors and solves the resulting optimization
problem by quadratic programming. The method is proposed in 
[Castaño et al.'s (2024)](https://ieeexplore.ieee.org/document/9791435/) paper.

In QuaPy, `EDy` works for binary and
multiclass problems and lets the user choose the pairwise distance through the
`distance` parameter (`'manhattan'`, `'euclidean'`, or a custom callable).
Because the optimization relies on `quadprog`, this method requires the
optional dependency `pip install quadprog`.

#### SMM

QuaPy also includes the binary `SMM` method of
[Hassan et al. (2019)](https://ieeexplore.ieee.org/document/9260028),
available as `qp.method.aggregative.SMM`. This is a very lightweight
distribution-matching variant in which the posterior representation is reduced
to class-wise means rather than full histograms, making it conceptually close
to PACC.


#### Kernel Density Estimation methods (KDEy)

QuaPy provides implementations for the three variants
of KDE-based methods proposed in 
_[Moreo, A., González, P. and del Coz, J.J.. 
Kernel Density Estimation for Multiclass Quantification. 
Machine Learning. Vol 114 (92), 2025](https://link.springer.com/article/10.1007/s10994-024-06726-5)_
(a [preprint](https://arxiv.org/abs/2401.00490) is available online). 
The variants differ in the divergence metric to be minimized:

- KDEy-HD: minimizes the (squared) Hellinger Distance and solves the problem via a Monte Carlo approach
- KDEy-CS: minimizes the Cauchy-Schwarz divergence and solves the problem via a closed-form solution
- KDEy-ML: minimizes the Kullback-Leibler divergence and solves the problem via maximum-likelihood

These methods are specifically devised for multiclass problems (although they can tackle 
binary problems too). 

All KDE-based methods depend on the hyperparameter `bandwidth` of the kernel. Typical values
that can be explored in model selection range in [0.01, 0.25]. Previous experiments reveal the methods' performance
varies smoothly at small variations of this hyperparameter.

A Bayesian counterpart is available as well; see the
{ref}`Bayesian Quantification Methods section <manuals/methods:Bayesian Quantification Methods>`
for `BayesianKDEy`.


### Explicit Loss Minimization

The Explicit Loss Minimization (ELM) represent a family of methods
based on structured output learning, i.e., quantifiers relying on 
classifiers that have been optimized targeting a 
quantification-oriented evaluation measure.
The original methods are implemented in QuaPy as classify & count (CC) 
quantifiers that use Joachim's [SVMperf](https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html) 
as the underlying classifier, properly set to optimize for the desired loss.
 
In QuaPy, this can be more achieved by calling the functions:

* `newSVMQ`: returns the quantification method called SVM(Q) that optimizes for the metric _Q_ defined 
in [_Barranquero, J., Díez, J., and del Coz, J. J. (2015). Quantification-oriented learning based
on reliable classifiers. Pattern Recognition, 48(2):591–604._](https://www.sciencedirect.com/science/article/pii/S003132031400291X) 
* `newSVMKLD` and `newSVMNKLD`: returns the quantification method called SVM(KLD) and SVM(nKLD), standing for 
    Kullback-Leibler Divergence and Normalized Kullback-Leibler Divergence, as proposed in [_Esuli, A. and Sebastiani, F. (2015). 
    Optimizing text quantifiers for multivariate loss functions. 
    ACM Transactions on Knowledge Discovery and Data, 9(4):Article 27._](https://dl.acm.org/doi/abs/10.1145/2700406)
* `newSVMAE` and `newSVMRAE`: returns a quantification method called SVM(AE) and SVM(RAE) that optimizes for the (Mean) Absolute Error and for the
  (Mean) Relative Absolute Error, as first used by 
    [_Moreo, A. and Sebastiani, F. (2021). Tweet sentiment quantification: An experimental re-evaluation. PLOS ONE 17 (9), 1-23._](https://arxiv.org/abs/2011.02552)

the last two methods (SVM(AE) and SVM(RAE)) have been implemented in 
QuaPy in order to make available ELM variants for what nowadays
are considered the most well-behaved evaluation metrics in quantification.

#### Installing the SVMperf backend

These methods rely on Joachim's [SVMperf](https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html),
patched with quantification-oriented losses. QuaPy provides the script
[`prepare_svmperf.sh`](https://github.com/HLT-ISTI/QuaPy/blob/master/prepare_svmperf.sh),
which downloads the original sources, applies the patch, and compiles the
resulting binary. In practice, this amounts to running:

```sh
./prepare_svmperf.sh
```

This creates a directory `svm_perf_quantification/`. Once this is available,
you can point QuaPy to it with:

```python
qp.environ['SVMPERF_HOME'] = './svm_perf_quantification'
```

The patch extends the one originally released for
[Esuli and Sebastiani (2015)](https://dl.acm.org/doi/abs/10.1145/2700406)
and also covers the `Q`, `AE`, and `RAE` losses used by QuaPy's ELM wrappers.

All ELM methods are binary because `SVMperf` itself is binary. They can still
be wrapped in a one-vs-all scheme for single-label multiclass problems, though
this strategy is generally considered inappropriate under prior probability
shift. See the examples on
[explicit loss minimization](https://github.com/HLT-ISTI/QuaPy/blob/devel/examples/17.explicit_loss_minimization.py)
and on
[one versus all quantification](https://github.com/HLT-ISTI/QuaPy/blob/devel/examples/10.one_vs_all.py)
for minimal working code.

## Non-Aggregative Methods

Non-aggregative methods are quantifiers that do not follow the two-step
(classify, then aggregate) pattern described above for aggregative methods.
These methods are implemented in the `qp.method.non_aggregative` module and
extend `BaseQuantifier` directly, implementing `fit` and `predict` on their own terms.

### Maximum Likelihood Prevalence Estimation (MLPE)

`MaximumLikelihoodPrevalenceEstimation` (MLPE) is a lazy baseline quantifier
that assumes the IID assumption holds, i.e., that there is no prior probability
shift between the training and the test distributions. Its `fit` method simply
computes and stores the training prevalence, and its `predict` method returns
that same training prevalence for any test sample, irrespective of the sample
itself. MLPE is considered a lower-bound quantifier: any quantification method
worth using should outperform it.

```python
import quapy as qp
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation

dataset = qp.datasets.fetch_UCIBinaryDataset('haberman')
train, test = dataset.train_test

model = MaximumLikelihoodPrevalenceEstimation()
model.fit(*train.Xy)
estim_prevalence = model.predict(test.X)  # always equals train.prevalence()
```

### Distribution Matching x (DMx) and Hellinger Distance x (HDx)

`DMx` is the covariate-space counterpart of the `DMy` distribution-matching
quantifier described in {ref}`the Hellinger Distance y (HDy) section <manuals/methods:Hellinger Distance y (HDy)>`:
instead of matching distributions built from the classifier's predictions, `DMx` matches
distributions built directly from the (discretized) feature space, and thus
requires no classifier at all. For each class, `DMx` builds one histogram per
feature from the training instances of that class; at prediction time, it
searches for the mixture of these class-conditional histograms that best
matches the (also histogram-based) representation of the test sample, in
terms of a chosen divergence.

`DMx` accepts the following hyperparameters in its constructor:

* `nbins`: the number of bins used to discretize each feature (default 8)
* `divergence`: a string ("HD" for Hellinger Distance, or "topsoe") or a
  callable taking two histograms and returning a divergence value (default "HD")
* `cdf`: whether to match cumulative distributions (CDFs) instead of the
  histograms (PDFs) themselves (default False)
* `search`: the strategy used for finding the optimal prevalence; valid
  options are `optim_minimize` (default, works for binary and multiclass
  problems), `linear_search`, and `ternary_search` (these last two are
  binary-only)
* `n_jobs`: number of parallel workers (default None)

`DMx` also offers the class method `DMx.HDx` (aliased as
`qp.method.non_aggregative.HDx`, and also as `HellingerDistanceX`) that
reproduces the original Hellinger Distance x (HDx) method proposed by
[González-Castro, Alaiz-Rodríguez, and Alegre (2013)](https://www.sciencedirect.com/science/article/pii/S0020025512004069),
the same paper that introduced HDy. HDx is a binary-only method that computes
the matching for `nbins` ranging over `[10, 20, ..., 110]` (via a
`MedianEstimator`, taking the median of the resulting estimates) and searches
for the best prevalence via a linear search stepping by 0.01, rather than via
the `optim_minimize` search used by `DMx` by default.

The following code, adapted from the example comparing HDy and HDx
(`examples/11.comparing_HDy_HDx.py`), shows the two methods side-by-side:

```python
from sklearn.linear_model import LogisticRegression
import quapy as qp
from quapy.method.aggregative import HDy
from quapy.method.non_aggregative import DMx

train, test = qp.datasets.fetch_UCIBinaryDataset('haberman').train_test
Xtr, ytr = train.Xy

hdy = HDy(LogisticRegression()).fit(Xtr, ytr)
estim_prevalence_hdy = hdy.predict(test.X)

hdx = DMx.HDx(n_jobs=-1).fit(Xtr, ytr)
estim_prevalence_hdx = hdx.predict(test.X)
```

Note that, unlike HDy, HDx requires no classifier whatsoever, since it
operates directly on the covariates.

### Energy Distance x (EDx)

QuaPy also provides `qp.method.non_aggregative.EDx`, which is the
feature-space counterpart of `EDy`: it keeps the same energy-distance
formulation and quadratic-program solver, but applies them directly to the raw
instances instead of first projecting them onto posterior probabilities through
a classifier. In this sense, `EDx` is to `EDy` what `DMx` is to `DMy`.

`EDx` works for binary and multiclass problems, accepts the same `distance`
options as `EDy` (`'manhattan'`, `'euclidean'`, or a custom callable), and
requires the optional dependency `pip install quadprog`.

### ReadMe

`ReadMe` is a non-aggregative quantification method proposed by
[Hopkins, D. and King, G. (2007). A method of automated nonparametric content
analysis for social science. American Journal of Political Science,
54(1):229-247.](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-5907.2009.00428.x)
The method estimates `Q(Y=i)` directly from `Q(X) = sum_i Q(X|Y=i) Q(Y=i)` by
solving a (constrained) least-squares regression, thus avoiding the cost of
estimating posterior probabilities `Q(Y=i|X)` altogether.

Since `Q(X)` and `Q(X|Y=i)` can be of very high dimension for realistic
feature spaces, ReadMe renders the problem tractable by performing bagging in
the feature space: many small random subsets of features (of size
`bagging_range`) are drawn, the least-squares problem is solved on each
subset, and the resulting estimates are averaged. ReadMe additionally
combines this bagging procedure with bootstrap resampling of the training
instances in order to derive confidence regions around the point estimate;
accordingly, `ReadMe` implements the `WithConfidenceABC` interface (see the
{ref}`confidence regions section <confidence-regions-for-class-prevalence-estimation>`),
and exposes a `predict_conf` method in addition to `predict`.

`ReadMe` accepts the following hyperparameters:

* `prob_model`: either `"full"` (default), the original Hopkins and King
  formulation, in which `Q(X)` and `Q(X|Y)` are modelled empirically and thus
  require the feature matrix `X` to be binary (e.g., term presence/absence);
  or `"naive"`, a much faster approximation that models `Q(X)` and `Q(X|Y)` as
  multinomial (bag-of-words) distributions, and that supports much larger
  values of `bagging_range`
* `bootstrap_trials`: number of bootstrap resamplings of the training data
  used for deriving the confidence region (default 300)
* `bagging_trials`: number of bagging trials, i.e., random feature subsets,
  averaged for each point estimate (default 300)
* `bagging_range`: number of features kept in each bagging trial (default 15);
  note that, when `prob_model="full"`, this value should typically be kept
  small (the authors advise against values above 25) since the empirical
  distribution requires enumerating `2^bagging_range` possible feature
  configurations
* `confidence_level`: the confidence level for the confidence region
  (default 0.95)
* `region`: the type of confidence region to construct, one of `"intervals"`
  (default), `"ellipse"`, `"ellipse-clr"`, or `"ellipse-ilr"` (see the
  {ref}`confidence regions section <confidence-regions-for-class-prevalence-estimation>`
  for details)
* `bonferroni`: whether to apply Bonferroni correction when `region="intervals"`
  (default `False`); this parameter has no effect for ellipse-based regions
* `random_state`: an int for replicability, or `None` (default)
* `verbose`: whether to display progress information (default False)

The following minimal example, adapted from
`examples/18.ReadMe_for_text_analysis.py`, shows ReadMe applied to a binary
bag-of-words text quantification problem:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import quapy as qp
from quapy.method.non_aggregative import ReadMe

reviews = qp.datasets.fetch_reviews('imdb').reduce(n_train=1000, random_state=0)

# ReadMe's "full" model requires a binary feature matrix
encode_0_1 = Pipeline([('0_1_terms', CountVectorizer(min_df=5, binary=True))])
train, test = qp.data.preprocessing.instance_transformation(reviews, encode_0_1, inplace=True).train_test

model = ReadMe(prob_model='full', bootstrap_trials=100, bagging_trials=100, bagging_range=20, random_state=0)
model.fit(*train.Xy)  # lazy: only bootstrap resampling happens here

estim_prevalence, conf_region = model.predict_conf(test.X)
```

Note that `ReadMe` is computationally expensive: its cost scales with the
product of `bootstrap_trials` and `bagging_trials`, each of which requires
solving a least-squares problem.


## Composable Methods

The `quapy.method.composable` module integrates [qunfold](https://github.com/mirkobunse/qunfold) allows the composition
of quantification methods from loss functions and feature transformations (thanks to Mirko Bunse for the integration!). 

Any composed method solves a linear system of equations by minimizing the loss after transforming the data. Methods of this kind include ACC, PACC, HDx, HDy, and many other well-known methods, as well as an unlimited number of re-combinations of their building blocks.

### Installation

```sh
pip install --upgrade pip setuptools wheel
pip install "jax[cpu]"
pip install "qunfold @ git+https://github.com/mirkobunse/qunfold@v0.1.5"
```

**Note:** since version 0.2.0, QuaPy is only compatible with qunfold >=0.1.5. 

### Basics

The composition of a method is implemented through the [](quapy.method.composable.ComposableQuantifier) class. Its documentation also features an example to get you started in composing your own methods.

```python
from quapy.method.composable import (
    ComposableQuantifier,
    TikhonovRegularized,
    LeastSquaresLoss,
    ClassRepresentation,
)

ComposableQuantifier( # ordinal ACC, as proposed by Bunse et al., 2022
    TikhonovRegularized(LeastSquaresLoss(), 0.01),
    ClassRepresentation(RandomForestClassifier(oob_score=True))
)
```

More exhaustive examples of method compositions, including hyper-parameter optimization, can be found in [the example directory](https://github.com/HLT-ISTI/QuaPy/tree/master/examples).

To implement your own loss functions and feature representations, follow the corresponding manual of the [qunfold package](https://github.com/mirkobunse/qunfold), which provides the back-end of QuaPy's composable module.

### Loss functions

- [](quapy.method.composable.LeastSquaresLoss)
- [](quapy.method.composable.EnergyLoss)
- [](quapy.method.composable.HellingerSurrogateLoss)
- [](quapy.method.composable.BlobelLoss)
- [](quapy.method.composable.CombinedLoss)

```{hint}
You can use the [](quapy.method.composable.CombinedLoss) to create arbitrary, weighted sums of losses and regularizers.
```

### Regularization functions

- [](quapy.method.composable.TikhonovRegularized)
- [](quapy.method.composable.TikhonovRegularization)

### Feature transformations

- [](quapy.method.composable.ClassRepresentation)
- [](quapy.method.composable.DistanceRepresentation)
- [](quapy.method.composable.HistogramRepresentation)
- [](quapy.method.composable.EnergyKernelRepresentation)
- [](quapy.method.composable.GaussianKernelRepresentation)
- [](quapy.method.composable.LaplacianKernelRepresentation)
- [](quapy.method.composable.GaussianRFFKernelRepresentation)

```{hint}
The [](quapy.method.composable.ClassRepresentation) requires the classifier to have a property `oob_score==True` and to produce a property `oob_decision_function` during fitting. In [scikit-learn](https://scikit-learn.org/), this requirement is fulfilled by any bagging classifier, such as random forests. Any other classifier needs to be cross-validated through the [](quapy.method.composable.CVClassifier).
```


## Meta Models

By _meta_ models we mean quantification methods that are defined on top of other
quantification methods, and that thus do not squarely belong to the aggregative nor
the non-aggregative group (indeed, _meta_ models could use quantifiers from any of those
groups).
_Meta_ models are implemented in the `qp.method.meta` module.

### Ensembles

QuaPy implements (some of) the variants proposed in:

* [_Pérez-Gállego, P., Quevedo, J. R., & del Coz, J. J. (2017).
Using ensembles for problems with characterizable changes in data distribution: A case study on quantification.
Information Fusion, 34, 87-100._](https://www.sciencedirect.com/science/article/pii/S1566253516300628)
* [_Pérez-Gállego, P., Castano, A., Quevedo, J. R., & del Coz, J. J. (2019). 
    Dynamic ensemble selection for quantification tasks. 
    Information Fusion, 45, 1-15._](https://www.sciencedirect.com/science/article/pii/S1566253517303652)

The following code shows how to instantiate an Ensemble of 30 _Adjusted Classify & Count_ (ACC) 
quantifiers operating with a _Logistic Regressor_ (LR) as the base classifier, and using the
_average_ as the aggregation policy (see the original article for further details).
The last parameter indicates to use all processors for parallelization.

```python
import quapy as qp
from quapy.method.aggregative import ACC
from quapy.method.meta import Ensemble
from sklearn.linear_model import LogisticRegression

dataset = qp.datasets.fetch_UCIBinaryDataset('haberman')
train, test = dataset.train_test

model = Ensemble(quantifier=ACC(LogisticRegression()), size=30, policy='ave', n_jobs=-1)
model.fit(*train.Xy)
estim_prevalence = model.predict(test.X)
```

Other aggregation policies implemented in QuaPy include:
* 'ptr' for applying a dynamic selection based on the training prevalence of the ensemble's members
* 'ds' for applying a dynamic selection based on the Hellinger Distance
* _any valid quantification measure_ (e.g., 'mse') for performing a static selection based on
the performance estimated for each member of the ensemble in terms of that evaluation metric.
  
When using any of the above options, it is important to set the `red_size` parameter, which 
informs of the number of members to retain.

Please, check the [model selection manual](./model-selection) if you want to optimize the hyperparameters of ensemble for classification or quantification.

### The QuaNet neural network

QuaPy offers an implementation of QuaNet, a deep learning model presented in:

[_Esuli, A., Moreo, A., & Sebastiani, F. (2018, October). 
A recurrent neural network for sentiment quantification. 
In Proceedings of the 27th ACM International Conference on 
Information and Knowledge Management (pp. 1775-1778)._](https://dl.acm.org/doi/abs/10.1145/3269206.3269287)

This model requires `torch` to be installed. 
QuaNet also requires a classifier that can provide embedded representations
of the inputs. 
In the original paper, QuaNet was tested using an LSTM as the base classifier.
In the following example, we show an instantiation of QuaNet that instead uses CNN as a probabilistic classifier, taking its last layer representation as the document embedding:

```python
import quapy as qp
from quapy.method.meta import QuaNet
from quapy.classification.neural import NeuralClassifierTrainer, CNNnet

# use samples of 100 elements
qp.environ['SAMPLE_SIZE'] = 100

# load the kindle dataset as text, and convert words to numerical indexes
dataset = qp.datasets.fetch_reviews('kindle', pickle=True)
qp.data.preprocessing.index(dataset, min_df=5, inplace=True)

# the text classifier is a CNN trained by NeuralClassifierTrainer
cnn = CNNnet(dataset.vocabulary_size, dataset.n_classes)
learner = NeuralClassifierTrainer(cnn, device='cuda')

# train QuaNet
model = QuaNet(learner, device='cuda')
model.fit(*dataset.training.Xy)
estim_prevalence = model.predict(dataset.test.X)
```

### HistNetQ

QuaPy offers an implementation of HistNetQ, a deep learning model based on a differentiable
histogram representation, presented in:

[_Pérez-Mon, O., Moreo, A., del Coz, J.J., & González, P. (2025).
Quantification using permutation-invariant networks based on histograms.
Neural Computing and Applications, 37(5), 3505-3520._](https://doi.org/10.1007/s00521-024-10721-1)

This model requires `torch` to be installed. Like QuaNet, HistNetQ is trained end-to-end on
samples ("bags") of known prevalence rather than on individually labeled instances; unlike QuaNet,
it requires no classifier at all, only an optional feature extraction module (a plain identity
module is used by default, for already-vectorized data).

```python
import quapy as qp
from quapy.method.meta import HistNetQ

dataset = qp.datasets.fetch_UCIBinaryDataset('haberman')

model = HistNetQ(bag_size=100, device='cpu')
model.fit(*dataset.training.Xy)
estim_prevalence = model.predict(dataset.test.X)
```

HistNetQ can alternatively be trained directly from a protocol that already provides the training
samples (e.g., when only bag-level prevalence values are available), via the `fit_from_samples`
method; see the API documentation for further details.


## Quantifiers with Uncertainty Quantification

_(New in v0.2.0!)_ Some quantification methods go beyond providing a single point estimate of class prevalence values and also produce confidence regions, which characterize the uncertainty around the point estimate. In QuaPy, two such families are currently implemented: bootstrap methods and Bayesian methods.

Confidence regions are constructed around a point estimate, which is typically computed as the mean value of a set of samples.

The confidence region can be instantiated in four ways:
* Confidence intervals: are standard confidence intervals generated for each class independently (_method="intervals"_). Since confidence intervals are independently derived for each class, Bonferroni correction can be applied.
* Confidence ellipse in the simplex: an ellipse constructed around the mean point; the ellipse lies on the simplex and takes
  into account possible inter-class dependencies in the data (_method="ellipse"_).
* Confidence ellipse in the Centered-Log Ratio (CLR) space: the underlying assumption of the ellipse is that the components are
  normally distributed. However, we know elements from the simplex have an inner structure. A better approach is to first
  transform the components into an unconstrained space (the CLR), and then construct the ellipse in such space (_method="ellipse-clr"_).
* Confidence ellipse in the Isometric-Log Ratio (ILR) space: analogous to the CLR-based ellipse, but built in the ILR space
  (_method="ellipse-ilr"_).

### Aggregative Bootstrap

The Aggregative Bootstrap method extends any aggregative quantifier by generating confidence regions for class prevalence estimates through bootstrapping. The method is described in the paper [Moreo, A., Salvati, N.
    An Efficient Method for Deriving Confidence Intervals in Aggregative Quantification.
    Learning to Quantify: Methods and Applications (LQ 2025), co-located at ECML-PKDD 2025.
    pp 12-33, Porto (Portugal)](https://lq-2025.github.io/proceedings/CompleteVolume.pdf). 
    
This implementation is optimized for aggregative quantifiers. The bootstrap is applied to pre-classified instances, significantly speeding up training and inference.
During training, bootstrap repetitions are performed only after training the classifier once. These repetitions are used to train multiple aggregation functions.
During inference, bootstrap is applied over pre-classified test instances.
  
Aggregative Bootstrap can be applied to any aggregative quantifier. For further information, check the [example](https://github.com/HLT-ISTI/QuaPy/tree/master/examples/16.confidence_regions.py) provided. A minimal working example is:

```python
import quapy as qp
from quapy.method.aggregative import PACC
from quapy.method.confidence import AggregativeBootstrap

train, test = qp.datasets.fetch_UCIMulticlassDataset('molecular').train_test

model = AggregativeBootstrap(
    PACC(),
    n_test_samples=200,
    confidence_level=0.95,
    region='ellipse-clr',  # choose among: intervals, ellipse, ellipse-clr, ellipse-ilr
    random_state=0,
)
model.fit(*train.Xy)
point_estimate, conf_region = model.predict_conf(test.X)
```

Here `region` makes the type of uncertainty region explicit. In practice, `intervals` is often the simplest default, while `ellipse`, `ellipse-clr`, and `ellipse-ilr` provide coupled regions over the simplex. If `region='intervals'`, you can additionally set `bonferroni=True` to apply Bonferroni correction; this flag has no effect for ellipse-based regions.

Beyond aggregative quantifiers, Bootstrap sampling can be applied to any type of quantification method, although
the speedup procedure described above is not applied.





### Bayesian Quantification Methods

QuaPy also provides a number of Bayesian quantifiers. While these methods are
usually related to an existing point-estimation family (e.g., ACC, EMQ/MLLS,
HDy, or KDEy), they differ enough in their goals and outputs to deserve a
separate presentation. In particular, Bayesian quantifiers typically return a
posterior mean rather than a single optimization result, expose posterior
samples or confidence regions, and often require additional probabilistic
inference dependencies.

The optional dependencies needed for these methods can be installed with:

```sh
pip install quapy[bayes]
```

#### BayesianCC (a Bayesian implementation of ACC)

The `BayesianCC` is a variant of ACC introduced in 
[Ziegler, A. and Czyż, P. "Bayesian quantification with black-box estimators", arXiv (2023)](https://arxiv.org/abs/2302.09159), 
which models the probabilities `q = Mp` using latent random variables with weak Bayesian priors, rather than 
plug-in probability estimates. In particular, it uses Markov Chain Monte Carlo sampling to find the values of 
`p` compatible with the observed quantities.
The `aggregate` method returns the posterior mean and the `get_prevalence_samples` method can be used to find 
uncertainty around `p` estimates (conditional on the observed data and the trained classifier) 
and is suitable for problems in which the `q = Mp` matrix is nearly non-invertible.

Note that this quantification method requires `val_split` to be a `float` and installation of additional dependencies (`$ pip install quapy[bayes]`) needed to run Markov chain Monte Carlo sampling. Markov Chain Monte Carlo is is slower than matrix inversion methods, but is guaranteed to sample proper probability vectors, so no clipping strategies are required.
An example presenting how to run the method and use posterior samples is available in `examples/bayesian_quantification.py`.

#### BayesianMAPLS (a Bayesian implementation of EMQ/MLLS)

`BayesianMAPLS` is a Bayesian variant of EMQ/MLLS proposed by
Ye, C. et al. (2024). Label shift estimation for class-imbalance problem: A
Bayesian approach. Proceedings of the IEEE/CVF Winter Conference on
Applications of Computer Vision (WACV 2024). QuaPy's implementation is
adapted from the [authors' reference code](https://github.com/ChangkunYe/MAPLS/blob/main/label_shift/mapls.py).
Rather than returning a single point estimate for the class prevalence, it
places a Dirichlet prior over the sought prevalence vector (in an
unconstrained, Isometric-Log-Ratio-transformed space) and samples from the
resulting posterior via Markov Chain Monte Carlo (using `numpyro`/`jax`),
conditioned on a preliminary MAP estimate obtained via the underlying `mapls`
routine. Like `BayesianCC`, its `aggregate` method returns the posterior mean,
while `predict_conf` additionally returns a confidence region (`intervals`,
`ellipse`, `ellipse-clr`, or `ellipse-ilr`) built from the posterior samples.
For interval regions, all Bayesian methods also accept `bonferroni=True` to
apply Bonferroni correction; this flag has no effect for ellipse-based regions.

This method requires installation of additional dependencies
(`$ pip install quapy[bayes]`) needed to run MCMC sampling; parameters
`num_warmup` and `num_samples` control the length of the chain, and `prior`
allows choosing between a uniform Dirichlet prior (default) or one of the
data-dependent priors ("map"/"map2") proposed in the original paper.

```python
import quapy as qp
from quapy.method._bayesian import BayesianMAPLS
from sklearn.linear_model import LogisticRegression

train, test = qp.datasets.fetch_UCIBinaryDataset('haberman').train_test

model = BayesianMAPLS(LogisticRegression())
model.fit(*train.Xy)
estim_prevalence, conf_region = model.predict_conf(test.X)
```

#### PQ: Precise Quantifier (a Bayesian implementation of HDy)

`PQ` (Precise Quantifier), available at `qp.method.confidence.PQ`, is a
Bayesian distribution-matching variant of `HDy` proposed in
[Igiraneza, A.B., Fraser, C., and Hinch, R. (2025). Estimating prevalence
with precision and accuracy.](https://arxiv.org/abs/2507.06061)
Rather than matching a single test histogram against a mixture of two
class-conditional histograms via a divergence measure (as `HDy` does), `PQ`
places the histogram-matching problem in a Bayesian setting and samples the
full posterior distribution over the (binary) prevalence value via Markov
Chain Monte Carlo (using `stan`). Its `aggregate` method returns the
posterior mean, while `predict_conf` additionally returns a confidence
region built from the posterior samples (`intervals`, `ellipse`, or
`ellipse-clr`).

`PQ` accepts `nbins` (the number of histogram bins, quantile-based by default,
or uniform if `fixed_bins=True`), and the usual MCMC controls `num_warmup`,
`num_samples`, and `stan_seed`. This method relies on the optional `stan`
dependency, installed via `$ pip install quapy[bayes]`.

```python
import quapy as qp
from quapy.method.confidence import PQ
from sklearn.linear_model import LogisticRegression

train, test = qp.datasets.fetch_UCIBinaryDataset('haberman').train_test

model = PQ(LogisticRegression())
model.fit(*train.Xy)
estim_prevalence, conf_region = model.predict_conf(test.X)
```

#### BayesianKDEy (a Bayesian implementation of KDEyML)

`BayesianKDEy`, available at `qp.method._bayesian.BayesianKDEy`, is a Bayesian
version of KDEy proposed by [Moreo et al. 2026](https://arxiv.org/abs/2607.04977).
Instead of solving for the single prevalence vector that
minimizes a divergence between the test distribution and a KDE-based mixture
model (as the KDEy variants above do), `BayesianKDEy` places a Dirichlet
prior over the prevalence vector and samples its posterior via Markov Chain
Monte Carlo (using `numpyro`/`jax`), conditioned on the same KDE mixture
components. Its `aggregate` method returns the posterior mean, while
`predict_conf` additionally returns a confidence region built from the
posterior samples.

In addition to the `kernel` and `bandwidth` hyperparameters (with the same
`gaussian`/`aitchison`/`ilr` kernel choice, and `shrinkage` regularization
for the latter two, available in `KDEyML`), `BayesianKDEy` exposes the usual
MCMC controls: `num_warmup`, `num_samples`, `mcmc_seed`, a `temperature` for
posterior calibration, and `prior` for choosing the Dirichlet prior
(`'uniform'` by default, or a custom scalar/array). This method relies on
the optional MCMC dependencies, installed via
`$ pip install quapy[bayes]`.

```python
import quapy as qp
from quapy.method._bayesian import BayesianKDEy
from sklearn.linear_model import LogisticRegression

train, test = qp.datasets.fetch_UCIBinaryDataset('haberman').train_test

model = BayesianKDEy(LogisticRegression(), bandwidth=0.1)
model.fit(*train.Xy)
estim_prevalence, conf_region = model.predict_conf(test.X)
```

