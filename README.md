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

```python
import quapy as qp
from sklearn.linear_model import LogisticRegression

dataset = qp.datasets.fetch_twitter('semeval16')

# create an "Adjusted Classify & Count" quantifier
model = qp.method.aggregative.ACC(LogisticRegression())
model.fit(dataset.training)

prevalences_estim = model.quantify(dataset.test.instances)
prevalences_true  = dataset.test.prevalence()

error = qp.error.mae(prevalences_true, prevalences_estim)

print(f'MAE={error:.3f}')
```

binary, and single-label


