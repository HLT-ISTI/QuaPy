import numpy
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import quapy as qp
from quapy.method.base import BinaryQuantifier
from quapy.data import Dataset, LabelledCollection
from quapy.method import AGGREGATIVE_METHODS, NON_AGGREGATIVE_METHODS
from quapy.method.aggregative import ACC, PACC, HDy
from quapy.method.meta import Ensemble

datasets = [pytest.param(qp.datasets.fetch_twitter('hcr', pickle=True), id='hcr'),
            pytest.param(qp.datasets.fetch_UCIDataset('ionosphere'), id='ionosphere')]

tinydatasets = [pytest.param(qp.datasets.fetch_twitter('hcr', pickle=True).reduce(), id='tiny_hcr'),
            pytest.param(qp.datasets.fetch_UCIDataset('ionosphere').reduce(), id='tiny_ionosphere')]

learners = [LogisticRegression, LinearSVC]


@pytest.mark.parametrize('dataset', datasets)
@pytest.mark.parametrize('aggregative_method', AGGREGATIVE_METHODS)
@pytest.mark.parametrize('learner', learners)
def test_aggregative_methods(dataset: Dataset, aggregative_method, learner):
    model = aggregative_method(learner())

    if isinstance(model, BinaryQuantifier) and not dataset.binary:
        print(f'skipping the test of binary model {type(model)} on non-binary dataset {dataset}')
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

    if isinstance(model, BinaryQuantifier) and not dataset.binary:
        print(f'skipping the test of binary model {model} on non-binary dataset {dataset}')
        return

    model.fit(dataset.training)

    estim_prevalences = model.quantify(dataset.test.instances)

    true_prevalences = dataset.test.prevalence()
    error = qp.error.mae(true_prevalences, estim_prevalences)

    assert type(error) == numpy.float64


@pytest.mark.parametrize('base_method', AGGREGATIVE_METHODS)
@pytest.mark.parametrize('learner', [LogisticRegression])
@pytest.mark.parametrize('dataset', tinydatasets)
@pytest.mark.parametrize('policy', Ensemble.VALID_POLICIES)
def test_ensemble_method(base_method, learner, dataset: Dataset, policy):
    qp.environ['SAMPLE_SIZE'] = 20
    base_quantifier=base_method(learner())
    if isinstance(base_quantifier, BinaryQuantifier) and not dataset.binary:
        print(f'skipping the test of binary model {base_quantifier} on non-binary dataset {dataset}')
        return
    if not dataset.binary and policy=='ds':
        print(f'skipping the test of binary policy ds on non-binary dataset {dataset}')
        return
    model = Ensemble(quantifier=base_quantifier, size=5, policy=policy, n_jobs=-1)

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


    qp.environ['SAMPLE_SIZE'] = 100

    # load the kindle dataset as text, and convert words to numerical indexes
    dataset = qp.datasets.fetch_reviews('kindle', pickle=True)
    dataset = Dataset(dataset.training.sampling(200, *dataset.training.prevalence()),
                      dataset.test.sampling(200, *dataset.test.prevalence()))
    qp.data.preprocessing.index(dataset, min_df=5, inplace=True)

    from quapy.classification.neural import CNNnet
    cnn = CNNnet(dataset.vocabulary_size, dataset.n_classes)

    from quapy.classification.neural import NeuralClassifierTrainer
    learner = NeuralClassifierTrainer(cnn, device='cuda')

    from quapy.method.meta import QuaNet
    model = QuaNet(learner, device='cuda')

    if isinstance(model, BinaryQuantifier) and not dataset.binary:
        print(f'skipping the test of binary model {model} on non-binary dataset {dataset}')
        return

    model.fit(dataset.training)

    estim_prevalences = model.quantify(dataset.test.instances)

    true_prevalences = dataset.test.prevalence()
    error = qp.error.mae(true_prevalences, estim_prevalences)

    assert type(error) == numpy.float64


def test_str_label_names():
    model = qp.method.aggregative.CC(LogisticRegression())

    dataset = qp.datasets.fetch_reviews('imdb', pickle=True)
    dataset = Dataset(dataset.training.sampling(1000, *dataset.training.prevalence()),
                      dataset.test.sampling(1000, 0.25, 0.75))
    qp.data.preprocessing.text2tfidf(dataset, min_df=5, inplace=True)

    numpy.random.seed(0)
    model.fit(dataset.training)

    int_estim_prevalences = model.quantify(dataset.test.instances)
    true_prevalences = dataset.test.prevalence()

    error = qp.error.mae(true_prevalences, int_estim_prevalences)
    assert type(error) == numpy.float64

    dataset_str = Dataset(LabelledCollection(dataset.training.instances,
                                             ['one' if label == 1 else 'zero' for label in dataset.training.labels]),
                          LabelledCollection(dataset.test.instances,
                                             ['one' if label == 1 else 'zero' for label in dataset.test.labels]))
    assert all(dataset_str.training.classes_ == dataset_str.test.classes_), 'wrong indexation'
    numpy.random.seed(0)
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
