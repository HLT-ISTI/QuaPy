import numpy
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import quapy as qp
from quapy.method import AGGREGATIVE_METHODS, NON_AGGREGATIVE_METHODS, EXPLICIT_LOSS_MINIMIZATION_METHODS
from quapy.method.meta import Ensemble

datasets = [pytest.param(qp.datasets.fetch_twitter('hcr'), id='hcr'),
            pytest.param(qp.datasets.fetch_UCIDataset('ionosphere'), id='ionosphere')]

learners = [LogisticRegression, MultinomialNB, LinearSVC]


@pytest.mark.parametrize('dataset', datasets)
@pytest.mark.parametrize('aggregative_method', AGGREGATIVE_METHODS.difference(EXPLICIT_LOSS_MINIMIZATION_METHODS))
@pytest.mark.parametrize('learner', learners)
def test_aggregative_methods(dataset, aggregative_method, learner):
    model = aggregative_method(learner())

    if model.binary and not dataset.binary:
        print(f'skipping the test of binary model {model} on non-binary dataset {dataset}')
        return

    model.fit(dataset.training)

    estim_prevalences = model.quantify(dataset.test.instances)

    true_prevalences = dataset.test.prevalence()
    error = qp.error.mae(true_prevalences, estim_prevalences)

    assert type(error) == numpy.float64


@pytest.mark.parametrize('dataset', datasets)
@pytest.mark.parametrize('elm_method', EXPLICIT_LOSS_MINIMIZATION_METHODS)
def test_elm_methods(dataset, elm_method):
    try:
        model = elm_method()
    except AssertionError as ae:
        if ae.args[0].find('does not seem to point to a valid path') > 0:
            print('Missing SVMperf binary program, skipping test')
            return

    if model.binary and not dataset.binary:
        print(f'skipping the test of binary model {model} on non-binary dataset {dataset}')
        return

    model.fit(dataset.training)

    estim_prevalences = model.quantify(dataset.test.instances)

    true_prevalences = dataset.test.prevalence()
    error = qp.error.mae(true_prevalences, estim_prevalences)

    assert type(error) == numpy.float64


@pytest.mark.parametrize('dataset', datasets)
@pytest.mark.parametrize('non_aggregative_method', NON_AGGREGATIVE_METHODS)
def test_non_aggregative_methods(dataset, non_aggregative_method):
    model = non_aggregative_method()

    if model.binary and not dataset.binary:
        print(f'skipping the test of binary model {model} on non-binary dataset {dataset}')
        return

    model.fit(dataset.training)

    estim_prevalences = model.quantify(dataset.test.instances)

    true_prevalences = dataset.test.prevalence()
    error = qp.error.mae(true_prevalences, estim_prevalences)

    assert type(error) == numpy.float64


@pytest.mark.parametrize('base_method', AGGREGATIVE_METHODS.difference(EXPLICIT_LOSS_MINIMIZATION_METHODS))
@pytest.mark.parametrize('learner', learners)
@pytest.mark.parametrize('dataset', datasets)
@pytest.mark.parametrize('policy', Ensemble.VALID_POLICIES)
def test_ensemble_method(base_method, learner, dataset, policy):
    qp.environ['SAMPLE_SIZE'] = len(dataset.training)
    model = Ensemble(quantifier=base_method(learner()), size=5, policy=policy, n_jobs=-1)
    if model.binary and not dataset.binary:
        print(f'skipping the test of binary model {model} on non-binary dataset {dataset}')
        return

    model.fit(dataset.training)

    estim_prevalences = model.quantify(dataset.test.instances)

    true_prevalences = dataset.test.prevalence()
    error = qp.error.mae(true_prevalences, estim_prevalences)

    assert type(error) == numpy.float64


def test_quanet_method():
    dataset = qp.datasets.fetch_reviews('kindle', pickle=True)
    qp.data.preprocessing.index(dataset, min_df=5, inplace=True)

    from quapy.classification.neural import CNNnet
    cnn = CNNnet(dataset.vocabulary_size, dataset.n_classes)

    from quapy.classification.neural import NeuralClassifierTrainer
    learner = NeuralClassifierTrainer(cnn, device='cuda')

    from quapy.method.meta import QuaNet
    model = QuaNet(learner, sample_size=len(dataset.training), device='cuda')

    if model.binary and not dataset.binary:
        print(f'skipping the test of binary model {model} on non-binary dataset {dataset}')
        return

    model.fit(dataset.training)

    estim_prevalences = model.quantify(dataset.test.instances)

    true_prevalences = dataset.test.prevalence()
    error = qp.error.mae(true_prevalences, estim_prevalences)

    assert type(error) == numpy.float64
