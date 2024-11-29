"""
This is a basic example showcasing some of the important concepts behind quapy.
First of all, import quapy. Wou would typically import quapy in the following way
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

import quapy as qp

# let's fetch some dataset to run one experiment
# datasets are available in the "qp.data.datasets" module (there is a shortcut in qp.datasets)

data = qp.datasets.fetch_reviews('hp')

# The data are in plain text format. You can convert them into tfidf using some utilities available in the
# qp.data.preprocessing module, e.g.:

data = qp.data.preprocessing.text2tfidf(data, min_df=5)

# you can obtain the same result by specifying tfidf=True it in the fetch function:
# data = qp.datasets.fetch_reviews('hp', tfidf=True, min_df=5)

# data is an object of type Dataset, a very basic collection that contains a "training" and a "test" collection inside.
train, test = data.train_test

# train and test are instances of LabelledCollection, a class that contains covariates (X) and true labels (y), along
# with sampling functionality. Here are some examples of usage:
X, y = train.Xy
print(f'number of classes {train.n_classes}')
print(f'class names {train.classes_}')

import quapy.functional as F  # <- this module has some functional utilities, like a string formatter for prevalences
print(f'training prevalence = {F.strprev(train.prevalence())}')

# let us train one quantifier, for example, PACC using a sklearn's Logistic Regressor as the underlying classifier
classifier = LogisticRegression()
pacc = qp.method.aggregative.PACC(classifier)

print(f'training {pacc}')
pacc.fit(train)

# let's now test our quantifier on the test data (of course, we should not use the test labels y at this point, only X)
X_test = test.X
estim_prevalence = pacc.quantify(X_test)

print(f'estimated test prevalence = {F.strprev(estim_prevalence)}')
print(f'true test prevalence = {F.strprev(test.prevalence())}')

# let us use some evaluation metric to check how well our quantifier fared.
# Error metrics are available in the qp.error module.

mae_error = qp.error.mae(test.prevalence(), estim_prevalence)
print(f'MAE={mae_error:.4f}')

# In quantification, we typically use an evaluation protocol to test the performance of a quantification method.
# The reason is that, even though the test set contains many instances, the whole counts as 1 single datapoint to
# the quantifier, because quantification targets samples of instances as a whole (while classification, or regression,
# target instances individually).
# Quapy provides some standard protocols in qp.protocol. We will use the artificial prevalence protocol (APP). APP
# works by generating many test samples, out of our original test collection, characterized by different prevalence
# values. To do so, a grid of prevalence values is explored, and different samples are generated conditioned on each
# prevalence vector. This way, the quantifier is stress-tested on a wide range of prevalence values, i.e., under
# prior probability shift conditions.

# In this case we use "test" and not only "test.X" since the protocol needs to know the class labels in order
# to generate samples at different prevalences. We will generate samples of 100 instances, from a grid of 21 values,
# i.e., from a grid = [0.0, 0.05, 0.10, ..., 1.00], and only one sample (repeats) for each combination.
app = qp.protocol.APP(test, sample_size=100, n_prevalences=21, repeats=1)

# let's print some examples:
show=5
for i, (sample, prev) in enumerate(app()):
    print(f'sample-{i}: {F.strprev(prev)}')
    if i+1==5:
        break

# we can use the evaluation routine provided in quapy to test our method using a given protocol in terms of
# one specific error metric
absolute_errors = qp.evaluation.evaluate(model=pacc, protocol=app, error_metric='ae')
print(f'MAE = {np.mean(absolute_errors):.4f}+-{np.std(absolute_errors):.4f}')





