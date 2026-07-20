```{toctree}
:hidden:

Home <self>
manuals
API <quapy>
```

# QuaPy

```{div} hero-copy
QuaPy is an open-source Python framework for quantification, also known as
supervised prevalence estimation or learning to quantify. It is designed with
research and experimental analysis in mind, and combines datasets, protocols,
evaluation measures, visualization tools, and a broad collection of
quantification methods in a single workflow.
```

`````{grid} 1 1 2 2
:gutter: 3
:class-container: landing-grid

````{grid-item-card} Quickstart
:class-card: landing-card
Install QuaPy and run your first quantifier in a few lines of code.
+++
```{button-link} #installation
:color: primary
Get Started
```
````

````{grid-item-card} Manuals
:class-card: landing-card
Hands-on guides with methodological context, literature pointers, and reproducible workflows.
+++
```{button-ref} manuals
:ref-type: doc
:color: primary
Open Manuals
```
````

````{grid-item-card} API
:class-card: landing-card
Browse the full reference for `quapy`, including methods, datasets, utilities, and research-oriented extensions.
+++
```{button-ref} quapy
:ref-type: doc
:color: primary
Browse API
```
````

````{grid-item-card} GitHub
:class-card: landing-card
Explore the source code, open issues, and current development branch activity.
+++
```{button-link} https://github.com/HLT-ISTI/QuaPy
:color: primary
Open GitHub
```
````

`````

## Installation

```sh
pip install quapy
```

## Why QuaPy

QuaPy is built around the concept of a data sample and supports the main tasks
in the quantification workflow: training quantifiers, generating evaluation
samples, measuring quantification error, selecting models under distribution
shift, and visualizing experimental behaviour. The framework is especially
suited for research settings, where one often needs not only implementations,
but also methodological context, literature links, and reproducible evaluation
procedures.

Some of the main features are:

* Implementation of many popular quantification methods, including Classify & Count and its variants,
  Expectation Maximization, HDy, QuaNet, quantification ensembles, and Bayesian extensions.
* Evaluation protocols for generating test samples under prior probability shift.
* A broad set of quantification-oriented evaluation metrics.
* Ready-to-use textual, numeric, and benchmark competition datasets.
* Method documentation that points back to the relevant literature and original papers.
* Native support for binary and single-label multiclass quantification.
* Visualization tools for analysing predictions, drift, confidence regions, and ternary prevalences.

## First Example

The following script fetches a binary dataset, trains an Adjusted Classify & Count quantifier,
and evaluates the resulting prevalence prediction with Mean Absolute Error.

```python
import quapy as qp

training, test = qp.datasets.fetch_UCIBinaryDataset("yeast").train_test

model = qp.method.aggregative.ACC()
Xtr, ytr = training.Xy
model.fit(Xtr, ytr)

estim_prevalence = model.predict(test.X)
true_prevalence = test.prevalence()

error = qp.error.mae(true_prevalence, estim_prevalence)
print(f'Mean Absolute Error (MAE)={error:.3f}')
```

Quantification is especially useful when the class prevalence of the test data
may differ from that of the training data. QuaPy implements protocols that make
it easy to evaluate methods across many such shifts. See the [](./manuals) for
worked examples.

## Citing QuaPy

If you find QuaPy useful, please consider citing the original paper.

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

If you want to contribute improvements to QuaPy, please open a pull request
against the `devel` branch.

## Acknowledgments

```{image} SoBigData.png
:width: 250px
:alt: SoBigData++
```

This work has been supported by the QuaDaSh project
_"Finanziato dall'Unione europea---Next Generation EU,
Missione 4 Componente 2 CUP B53D23026250001"_.
