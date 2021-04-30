import numpy
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import quapy as qp
from quapy.method import AGGREGATIVE_METHODS

datasets = [pytest.param(qp.datasets.fetch_twitter('hcr'), id='hcr'),
            pytest.param(qp.datasets.fetch_UCIDataset('ionosphere'), id='ionosphere')]

learners = [LogisticRegression, MultinomialNB, LinearSVC]


@pytest.mark.parametrize('dataset', datasets)
@pytest.mark.parametrize('aggregative_method', AGGREGATIVE_METHODS)
@pytest.mark.parametrize('learner', learners)
def test_aggregative_methods(dataset, aggregative_method, learner):
    model = aggregative_method(learner())

    if model.binary and not dataset.binary:
        return

    model.fit(dataset.training)

    estim_prevalences = model.quantify(dataset.test.instances)

    true_prevalences = dataset.test.prevalence()
    error = qp.error.mae(true_prevalences, estim_prevalences)

    assert type(error) == numpy.float64
