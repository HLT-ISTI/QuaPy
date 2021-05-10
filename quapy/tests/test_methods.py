import numpy
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import quapy as qp
from quapy.data import Dataset, LabelledCollection
from quapy.method import AGGREGATIVE_METHODS, NON_AGGREGATIVE_METHODS, EXPLICIT_LOSS_MINIMIZATION_METHODS
from quapy.method.aggregative import ACC, PACC, HDy
from quapy.method.meta import Ensemble

datasets = [pytest.param(qp.datasets.fetch_twitter('hcr'), id='hcr'),
            pytest.param(qp.datasets.fetch_UCIDataset('ionosphere'), id='ionosphere')]

learners = [LogisticRegression, LinearSVC]


@pytest.mark.parametrize('dataset', datasets)
@pytest.mark.parametrize('aggregative_method', AGGREGATIVE_METHODS.difference(EXPLICIT_LOSS_MINIMIZATION_METHODS))
@pytest.mark.parametrize('learner', learners)
def test_aggregative_methods(dataset: Dataset, aggregative_method, learner):
    model = aggregative_method(learner())

    if model.binary and not dataset.binary:
        print(f'skipping the test of binary model {type(model)} on non-binary dataset {dataset}')
        return

    model.fit(dataset.training)

    estim_prevalences = model.quantify(dataset.test.instances)

    true_prevalences = dataset.test.prevalence()
    error = qp.error.mae(true_prevalences, estim_prevalences)

    assert type(error) == numpy.float64


@pytest.mark.parametrize('dataset', datasets)
@pytest.mark.parametrize('elm_method', EXPLICIT_LOSS_MINIMIZATION_METHODS)
def test_elm_methods(dataset: Dataset, elm_method):
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
def test_non_aggregative_methods(dataset: Dataset, non_aggregative_method):
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
def test_ensemble_method(base_method, learner, dataset: Dataset, policy):
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
    try:
        import quapy.classification.neural
    except ModuleNotFoundError:
        print('skipping QuaNet test due to missing torch package')
        return

    dataset = qp.datasets.fetch_reviews('kindle', pickle=True)
    dataset = Dataset(dataset.training.sampling(100, *dataset.training.prevalence()),
                      dataset.test.sampling(100, *dataset.test.prevalence()))
    qp.data.preprocessing.index(dataset, min_df=5, inplace=True)

    from quapy.classification.neural import CNNnet
    cnn = CNNnet(dataset.vocabulary_size, dataset.training.n_classes)

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


def models_to_test_for_str_label_names():
    models = list()
    learner = LogisticRegression
    for method in AGGREGATIVE_METHODS.difference(EXPLICIT_LOSS_MINIMIZATION_METHODS):
        models.append(method(learner()))
    for method in NON_AGGREGATIVE_METHODS:
        models.append(method())
    return models


@pytest.mark.parametrize('model', models_to_test_for_str_label_names())
def test_str_label_names(model):
    if type(model) in {ACC, PACC, HDy}:
        print(
            f'skipping the test of binary model {type(model)} because it currently does not support random seed control.')
        return

    dataset = qp.datasets.fetch_reviews('imdb', pickle=True)
    dataset = Dataset(dataset.training.sampling(1000, *dataset.training.prevalence()),
                      dataset.test.sampling(1000, *dataset.test.prevalence()))
    qp.data.preprocessing.text2tfidf(dataset, min_df=5, inplace=True)

    model.fit(dataset.training)

    int_estim_prevalences = model.quantify(dataset.test.instances)
    true_prevalences = dataset.test.prevalence()

    error = qp.error.mae(true_prevalences, int_estim_prevalences)
    assert type(error) == numpy.float64

    dataset_str = Dataset(LabelledCollection(dataset.training.instances,
                                             ['one' if label == 1 else 'zero' for label in dataset.training.labels]),
                          LabelledCollection(dataset.test.instances,
                                             ['one' if label == 1 else 'zero' for label in dataset.test.labels]))

    model.fit(dataset_str.training)

    str_estim_prevalences = model.quantify(dataset_str.test.instances)
    true_prevalences = dataset_str.test.prevalence()

    error = qp.error.mae(true_prevalences, str_estim_prevalences)
    assert type(error) == numpy.float64

    print(true_prevalences)
    print(int_estim_prevalences)
    print(str_estim_prevalences)

    numpy.testing.assert_almost_equal(int_estim_prevalences[1],
                                      str_estim_prevalences[list(model.classes_).index('one')])
