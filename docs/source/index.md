```{toctree}
:hidden:

self
```

# Quickstart

QuaPy is an open source framework for quantification (a.k.a. supervised prevalence estimation, or learning to quantify) written in Python.

QuaPy is based on the concept of "data sample", and provides implementations of the most important aspects of the quantification workflow, such as (baseline and advanced) quantification methods, quantification-oriented model selection mechanisms, evaluation measures, and evaluations protocols used for evaluating quantification methods. QuaPy also makes available commonly used datasets, and offers visualization tools for facilitating the analysis and interpretation of the experimental results.

QuaPy is hosted on GitHub at [https://github.com/HLT-ISTI/QuaPy](https://github.com/HLT-ISTI/QuaPy).

## Installation

```sh
pip install quapy
```

## Usage

The following script fetches a dataset of tweets, trains, applies, and evaluates a quantifier based on the *Adjusted Classify & Count* quantification method, using, as the evaluation measure, the *Mean Absolute Error* (MAE) between the predicted and the true class prevalence values of the test set:

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

Quantification is useful in scenarios characterized by prior probability shift. In other words, we would be little interested in estimating the class prevalence values of the test set if we could assume the IID assumption to hold, as this prevalence would be roughly equivalent to the class prevalence of the training set. For this reason, any quantification model should be tested across many samples, even ones characterized by class prevalence values different or very different from those found in the training set. QuaPy implements sampling procedures and evaluation protocols that automate this workflow. See the [](./manuals) for detailed examples.

## Manuals

The following manuals illustrate several aspects of QuaPy through examples:

```{toctree}
:maxdepth: 3

manuals
```

```{toctree}
:hidden:

API <quapy>
```

## Features

- Implementation of many popular quantification methods (Classify-&-Count and its variants, Expectation Maximization, quantification methods based on structured output learning, HDy, QuaNet, quantification ensembles, among others).
- Versatile functionality for performing evaluation based on sampling generation protocols (e.g., APP, NPP, etc.).
- Implementation of most commonly used evaluation metrics (e.g., AE, RAE, NAE, NRAE, SE, KLD, NKLD, etc.).
- Datasets frequently used in quantification (textual and numeric), including:
    - 32 UCI Machine Learning binary datasets.
    - 5 UCI Machine Learning multiclass datasets (new in v0.1.8!).
    - 11 Twitter quantification-by-sentiment datasets.
    - 3 product reviews quantification-by-sentiment datasets.
    - 4 tasks from LeQua competition (new in v0.1.7!)
    - IFCB dataset of plankton water samples (new in v0.1.8!).
- Native support for binary and single-label multiclass quantification scenarios.
- Model selection functionality that minimizes quantification-oriented loss functions.
- Visualization tools for analysing the experimental results.

## Citing QuaPy

If you find QuaPy useful (and we hope you will), please consider citing the original paper in your research.

```bibtex
@inproceedings{moreo2021quapy,
   title={QuaPy: a python-based framework for quantification},
   author={Moreo, Alejandro and Esuli, Andrea and Sebastiani, Fabrizio},
   booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
   pages={4534--4543},
   year={2021}
}
```

## Contributing

In case you want to contribute improvements to quapy, please generate pull request to the "devel" branch.

## Acknowledgments

```{image} SoBigData.png
:width: 250px
:alt: SoBigData++
```
