# Quantification Methods

Quantification methods can be categorized as belonging to
`aggregative`, `non-aggregative`, and `meta-learning` groups. 
Most methods included in QuaPy at the moment are of type `aggregative`
(though we plan to add many more methods in the near future), i.e.,
are methods characterized by the fact that
quantification is performed as an aggregation function of the individual
products of classification.

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
 simply relies on the label predictions of a classifier to deliver class estimates.
* _ACC_ (Adjusted Classify & Count), the adjusted variant of CC.
* _PCC_ (Probabilistic Classify & Count), the probabilistic variant of CC that
relies on the soft estimations (or posterior probabilities) returned by a (probabilistic) classifier.
* _PACC_ (Probabilistic Adjusted Classify & Count), the adjusted variant of PCC.

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
estim_prevalence = model.predict(test.instances)
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


#### BayesianCC

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

### Expectation Maximization (EMQ)

The Expectation Maximization Quantifier (EMQ), also known as
the SLD, is available at `qp.method.aggregative.EMQ` or via the 
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


### Hellinger Distance y (HDy)

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

QuaPy also provides an implementation of the generalized
"Distribution Matching" approaches for multiclass, inspired by the framework
of [Firat (2016)](https://arxiv.org/abs/1606.00868). One can instantiate
a variant of HDy for multiclass quantification as follows:

```python
mutliclassHDy = qp.method.aggregative.DMy(classifier=LogisticRegression(), divergence='HD', cdf=False)
``` 

QuaPy also provides an implementation of the "DyS"
framework proposed by [Maletzke et al (2020)](https://ojs.aaai.org/index.php/AAAI/article/view/4376)
and the "SMM" method proposed by [Hassan et al (2019)](https://ieeexplore.ieee.org/document/9260028)
(thanks to _Pablo González_ for the contributions!)

### Threshold Optimization methods

QuaPy implements Forman's threshold optimization methods;
see, e.g., [(Forman 2006)](https://dl.acm.org/doi/abs/10.1145/1150402.1150423) 
and [(Forman 2008)](https://link.springer.com/article/10.1007/s10618-008-0097-y).
These include: `T50`, `MAX`, `X`, Median Sweep (`MS`), and its variant `MS2`.

These methods are binary-only and implement different heuristics for 
improving the stability of the denominator of the ACC adjustment (`tpr-fpr`).
The methods are called "threshold" since said heuristics have to do
with different choices of the underlying classifier's threshold.

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

In order to make these models work, you would need to run the script
`prepare_svmperf.sh` (distributed along with QuaPy) that
downloads `SVMperf`' source code, applies a patch that 
implements the quantification oriented losses, and compiles the
sources.

If you want to add any custom loss, you would need to modify
the source code of `SVMperf` in order to implement it, and
assign a valid loss code to it. Then you must re-compile 
the whole thing and instantiate the quantifier in QuaPy
as follows:

```python
# you can either set the path to your custom svm_perf_quantification implementation
# in the environment variable, or as an argument to the constructor of ELM
qp.environ['SVMPERF_HOME'] = './path/to/svm_perf_quantification'

# assign an alias to your custom loss and the id you have assigned to it
svmperf = qp.classification.svmperf.SVMperf
svmperf.valid_losses['mycustomloss'] = 28

# instantiate the ELM method indicating the loss
model = qp.method.aggregative.ELM(loss='mycustomloss')
```

All ELM are binary quantifiers since they rely on `SVMperf`, that
currently supports only binary classification.
ELM variants (any binary quantifier in general) can be extended
to operate in single-label scenarios trivially by adopting a 
"one-vs-all" strategy (as, e.g., in 
[_Gao, W. and Sebastiani, F. (2016). From classification to quantification in tweet sentiment
analysis. Social Network Analysis and Mining, 6(19):1–22_](https://link.springer.com/article/10.1007/s13278-016-0327-z)).
In QuaPy this is possible by using the `OneVsAll` class.

There are two ways for instantiating this class, `OneVsAllGeneric` that works for
any quantifier, and `OneVsAllAggregative` that is optimized for aggregative quantifiers.
In general, you can simply use the `newOneVsAll` function and QuaPy will choose
the more convenient of the two.

```python
import quapy as qp
from quapy.method.aggregative import SVMQ

# load a single-label dataset (this one contains 3 classes)
dataset = qp.datasets.fetch_twitter('hcr', pickle=True)

# let qp know where svmperf is
qp.environ['SVMPERF_HOME'] = '../svm_perf_quantification'

model = newOneVsAll(SVMQ(), n_jobs=-1)  # run them on parallel
model.fit(dataset.training)
estim_prevalence = model.predict(dataset.test.instances)
```

Check the examples on [explicit loss minimization](https://github.com/HLT-ISTI/QuaPy/blob/devel/examples/17.explicit_loss_minimization.py)
and on [one versus all quantification](https://github.com/HLT-ISTI/QuaPy/blob/devel/examples/10.one_vs_all.py) for more details.
**Note** that the _one versus all_ approach is considered inappropriate under prior probability shift, though. 

### Kernel Density Estimation methods (KDEy)

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
ComposableQuantifier( # ordinal ACC, as proposed by Bunse et al., 2022
  TikhonovRegularized(LeastSquaresLoss(), 0.01),
  ClassTransformer(RandomForestClassifier(oob_score=True))
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

- [](quapy.method.composable.ClassTransformer)
- [](quapy.method.composable.DistanceTransformer)
- [](quapy.method.composable.HistogramTransformer)
- [](quapy.method.composable.EnergyKernelTransformer)
- [](quapy.method.composable.GaussianKernelTransformer)
- [](quapy.method.composable.LaplacianKernelTransformer)
- [](quapy.method.composable.GaussianRFFKernelTransformer)

```{hint}
The [](quapy.method.composable.ClassTransformer) requires the classifier to have a property `oob_score==True` and to produce a property `oob_decision_function` during fitting. In [scikit-learn](https://scikit-learn.org/), this requirement is fulfilled by any bagging classifier, such as random forests. Any other classifier needs to be cross-validated through the [](quapy.method.composable.CVClassifier).
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

## Confidence Regions for Class Prevalence Estimation

_(New in v0.2.0!)_ Some quantification methods go beyond providing a single point estimate of class prevalence values and also produce confidence regions, which characterize the uncertainty around the point estimate. In QuaPy, two such methods are currently implemented:

* Aggregative Bootstrap: The Aggregative Bootstrap method extends any aggregative quantifier by generating confidence regions for class prevalence estimates through bootstrapping. Key features of this method include:

    * Optimized Computation: The bootstrap is applied to pre-classified instances, significantly speeding up training and inference.
During training, bootstrap repetitions are performed only after training the classifier once. These repetitions are used to train multiple aggregation functions.
During inference, bootstrap is applied over pre-classified test instances.
  * General Applicability: Aggregative Bootstrap can be applied to any aggregative quantifier.
  For further information, check the [example](https://github.com/HLT-ISTI/QuaPy/tree/master/examples/16.confidence_regions.py) provided.

* BayesianCC: is a Bayesian variant of the Adjusted Classify & Count (ACC) quantifier; see more details in the [example](https://github.com/HLT-ISTI/QuaPy/tree/master/examples/14.bayesian_quantification.py) provided.

Confidence regions are constructed around a point estimate, which is typically computed as the mean value of a set of samples.
The confidence region can be instantiated in three ways:
* Confidence intervals: are standard confidence intervals generated for each class independently (_method="intervals"_).
* Confidence ellipse in the simplex: an ellipse constructed around the mean point; the ellipse lies on the simplex and takes
  into account possible inter-class dependencies in the data (_method="ellipse"_). 
* Confidence ellipse in the Centered-Log Ratio (CLR) space: the underlying assumption of the ellipse is that the components are
  normally distributed. However, we know elements from the simplex have an inner structure. A better approach is to first
  transform the components into an unconstrained space (the CLR), and then construct the ellipse in such space (_method="ellipse-clr"_).