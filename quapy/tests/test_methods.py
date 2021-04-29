import numpy
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import quapy as qp

datasets = [qp.datasets.fetch_twitter('semeval16')]

aggregative_methods = [qp.method.aggregative.CC, qp.method.aggregative.ACC, qp.method.aggregative.ELM]

learners = [LogisticRegression, MultinomialNB, LinearSVC]


@pytest.mark.parametrize('dataset', datasets)
@pytest.mark.parametrize('aggregative_method', aggregative_methods)
@pytest.mark.parametrize('learner', learners)
def test_aggregative_methods(dataset, aggregative_method, learner):
    model = aggregative_method(learner())

    model.fit(dataset.training)

    estim_prevalences = model.quantify(dataset.test.instances)

    true_prevalences = dataset.test.prevalence()
    error = qp.error.mae(true_prevalences, estim_prevalences)

    assert type(error) == numpy.float64
