# QuaPy

QuaPy is an open source framework for quantification (a.k.a. supervised prevalence estimation, or learning to quantify)
written in Python.

QuaPy is based on the concept of "data sample", and provides implementations of the
most important aspects of the quantification workflow, such as (baseline and advanced)
quantification methods, 
quantification-oriented model selection mechanisms, evaluation measures, and evaluations protocols
used for evaluating quantification methods.
QuaPy also makes available commonly used datasets, and offers visualization tools 
for facilitating the analysis and interpretation of the experimental results.

### Last updates:

* A detailed documentation is now available [here](https://hlt-isti.github.io/QuaPy/)
* The developer API documentation is available [here](https://hlt-isti.github.io/QuaPy/build/html/modules.html)

### Installation

```commandline
pip install quapy
```

## A quick example:

The following script fetches a dataset of tweets, trains, applies, and evaluates a quantifier based on the 
_Adjusted Classify & Count_ quantification method, using, as the evaluation measure, the _Mean Absolute Error_ (MAE)
between the predicted and the true class prevalence values
of the test set.

```python
import quapy as qp
from sklearn.linear_model import LogisticRegression

dataset = qp.datasets.fetch_twitter('semeval16')

# create an "Adjusted Classify & Count" quantifier
model = qp.method.aggregative.ACC(LogisticRegression())
model.fit(dataset.training)

estim_prevalence = model.quantify(dataset.test.instances)
true_prevalence  = dataset.test.prevalence()

error = qp.error.mae(true_prevalence, estim_prevalence)

print(f'Mean Absolute Error (MAE)={error:.3f}')
```

Quantification is useful in scenarios characterized by prior probability shift. In other
words, we would be little interested in estimating the class prevalence values of the test set if 
we could assume the IID assumption to hold, as this prevalence would be roughly equivalent to the 
class prevalence of the training set. For this reason, any quantification model 
should be tested across many samples, even ones characterized by class prevalence 
values different or very different from those found in the training set.
QuaPy implements sampling procedures and evaluation protocols that automate this workflow.
See the [Wiki](https://github.com/HLT-ISTI/QuaPy/wiki) for detailed examples.

## Features

* Implementation of many popular quantification methods (Classify-&-Count and its variants, Expectation Maximization,
quantification methods based on structured output learning, HDy, QuaNet, and quantification ensembles).
* Versatile functionality for performing evaluation based on artificial sampling protocols.
* Implementation of most commonly used evaluation metrics (e.g., AE, RAE, SE, KLD, NKLD, etc.).
* Datasets frequently used in quantification (textual and numeric), including:
    * 32 UCI Machine Learning datasets.
    * 11 Twitter quantification-by-sentiment datasets.
    * 3 product reviews quantification-by-sentiment datasets. 
* Native support for binary and single-label multiclass quantification scenarios.
* Model selection functionality that minimizes quantification-oriented loss functions.
* Visualization tools for analysing the experimental results.

## Requirements

* scikit-learn, numpy, scipy
* pytorch (for QuaNet)
* svmperf patched for quantification (see below)
* joblib
* tqdm
* pandas, xlrd
* matplotlib

## SVM-perf with quantification-oriented losses
In order to run experiments involving SVM(Q), SVM(KLD), SVM(NKLD),
SVM(AE), or SVM(RAE), you have to first download the 
[svmperf](http://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html) 
package, apply the patch 
[svm-perf-quantification-ext.patch](./svm-perf-quantification-ext.patch), and compile the sources.
The script [prepare_svmperf.sh](prepare_svmperf.sh) does all the job. Simply run:

```
./prepare_svmperf.sh
```

The resulting directory [svm_perf_quantification](./svm_perf_quantification) contains the
patched version of _svmperf_ with quantification-oriented losses. 

The [svm-perf-quantification-ext.patch](./svm-perf-quantification-ext.patch) is an extension of the patch made available by
[Esuli et al. 2015](https://dl.acm.org/doi/abs/10.1145/2700406?casa_token=8D2fHsGCVn0AAAAA:ZfThYOvrzWxMGfZYlQW_y8Cagg-o_l6X_PcF09mdETQ4Tu7jK98mxFbGSXp9ZSO14JkUIYuDGFG0) 
that allows SVMperf to optimize for
the _Q_ measure as proposed by [Barranquero et al. 2015](https://www.sciencedirect.com/science/article/abs/pii/S003132031400291X) 
and for the _KLD_ and _NKLD_ measures as proposed by [Esuli et al. 2015](https://dl.acm.org/doi/abs/10.1145/2700406?casa_token=8D2fHsGCVn0AAAAA:ZfThYOvrzWxMGfZYlQW_y8Cagg-o_l6X_PcF09mdETQ4Tu7jK98mxFbGSXp9ZSO14JkUIYuDGFG0).
This patch extends the above one by also allowing SVMperf to optimize for 
_AE_ and _RAE_.
  
  
## Documentation

The [developer API documentation](https://hlt-isti.github.io/QuaPy/build/html/modules.html) is available [here](https://hlt-isti.github.io/QuaPy/build/html/index.html). 

Check out our [Wiki](https://github.com/HLT-ISTI/QuaPy/wiki), in which many examples
are provided:

* [Datasets](https://github.com/HLT-ISTI/QuaPy/wiki/Datasets)
* [Evaluation](https://github.com/HLT-ISTI/QuaPy/wiki/Evaluation)
* [Methods](https://github.com/HLT-ISTI/QuaPy/wiki/Methods)
* [Model Selection](https://github.com/HLT-ISTI/QuaPy/wiki/Model-Selection)
* [Plotting](https://github.com/HLT-ISTI/QuaPy/wiki/Plotting)
