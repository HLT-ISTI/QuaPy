# QuaPy

QuaPy is an open source framework for Quantification (a.k.a. Supervised Prevalence Estimation)
written in Python.

QuaPy roots on the concept of data sample, and provides implementations of
most important concepts in quantification literature, such as the most important 
quantification baselines, many advanced quantification methods, 
quantification-oriented model selection, many evaluation measures and protocols
used for evaluating quantification methods.
QuaPy also integrates commonly used datasets and offers visualization tools 
for facilitating the analysis and interpretation of results.

### Installation

```commandline
pip install quapy
```

## A quick example:

The following script fetchs a Twitter dataset, trains and evaluates an 
_Adjusted Classify & Count_ model in terms of the _Mean Absolute Error_ (MAE)
between the class prevalences estimated for the test set and the true prevalences
of the test set.

```python
import quapy as qp
from sklearn.linear_model import LogisticRegression

dataset = qp.datasets.fetch_twitter('semeval16')

# create an "Adjusted Classify & Count" quantifier
model = qp.method.aggregative.ACC(LogisticRegression())
model.fit(dataset.training)

estim_prevalences = model.quantify(dataset.test.instances)
true_prevalences  = dataset.test.prevalence()

error = qp.error.mae(true_prevalences, estim_prevalences)

print(f'Mean Absolute Error (MAE)={error:.3f}')
```

Quantification is useful in scenarios of prior probability shift. In other
words, we would not be interested in estimating the class prevalences of the test set if 
we could assume the IID assumption to hold, as this prevalence would simply coincide with the 
class prevalence of the training set. For this reason, any Quantification model 
should be tested across samples characterized by different class prevalences.
QuaPy implements sampling procedures and evaluation protocols that automates this endeavour.
See the [Wiki](https://github.com/HLT-ISTI/QuaPy/wiki) for detailed examples.

## Features

* Implementation of most popular quantification methods (Classify-&-Count variants, Expectation-Maximization,
SVM-based variants for quantification, HDy, QuaNet, and Ensembles).
* Versatile functionality for performing evaluation based on artificial sampling protocols.
* Implementation of most commonly used evaluation metrics (e.g., MAE, MRAE, MSE, NKLD, etc.).
* Popular datasets for Quantification (textual and numeric) available, including:
    * 32 UCI Machine Learning datasets.
    * 11 Twitter Sentiment datasets.
    * 3 Reviews Sentiment datasets. 
* Native supports for binary and single-label scenarios of quantification.
* Model selection functionality targeting quantification-oriented losses.
* Visualization tools for analysing results.

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
and for the _KLD_ and _NKLD_ as proposed by [Esuli et al. 2015](https://dl.acm.org/doi/abs/10.1145/2700406?casa_token=8D2fHsGCVn0AAAAA:ZfThYOvrzWxMGfZYlQW_y8Cagg-o_l6X_PcF09mdETQ4Tu7jK98mxFbGSXp9ZSO14JkUIYuDGFG0)
for quantification.
This patch extends the former by also allowing SVMperf to optimize for 
_AE_ and _RAE_.
  
  
## Wiki

Check out our [Wiki](https://github.com/HLT-ISTI/QuaPy/wiki) in which many examples
are provided:

* [Datasets](https://github.com/HLT-ISTI/QuaPy/wiki/Datasets)
* [Evaluation](https://github.com/HLT-ISTI/QuaPy/wiki/Evaluation)
* [Methods](https://github.com/HLT-ISTI/QuaPy/wiki/Methods)
* [Model Selection](https://github.com/HLT-ISTI/QuaPy/wiki/Model-Selection)
* [Plotting](https://github.com/HLT-ISTI/QuaPy/wiki/Plotting)