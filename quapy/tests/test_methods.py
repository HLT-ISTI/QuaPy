import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import method.aggregative
import quapy as qp
from quapy.model_selection import GridSearchQ
from quapy.method.base import BinaryQuantifier
from quapy.data import Dataset, LabelledCollection
from quapy.method import AGGREGATIVE_METHODS, NON_AGGREGATIVE_METHODS
from quapy.method.meta import Ensemble
from quapy.protocol import APP
from quapy.method.aggregative import DMy
from quapy.method.meta import MedianEstimator

# datasets = [pytest.param(qp.datasets.fetch_twitter('hcr', pickle=True), id='hcr'),
#             pytest.param(qp.datasets.fetch_UCIDataset('ionosphere'), id='ionosphere')]

tinydatasets = [pytest.param(qp.datasets.fetch_twitter('hcr', pickle=True).reduce(), id='tiny_hcr'),
                pytest.param(qp.datasets.fetch_UCIBinaryDataset('ionosphere').reduce(), id='tiny_ionosphere')]

learners = [LogisticRegression, LinearSVC]


@pytest.mark.parametrize('dataset', tinydatasets)
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

    assert type(error) == np.float64


@pytest.mark.parametrize('dataset', tinydatasets)
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

    assert type(error) == np.float64


@pytest.mark.parametrize('base_method', [method.aggregative.ACC, method.aggregative.PACC])
@pytest.mark.parametrize('learner', [LogisticRegression])
@pytest.mark.parametrize('dataset', tinydatasets)
@pytest.mark.parametrize('policy', Ensemble.VALID_POLICIES)
def test_ensemble_method(base_method, learner, dataset: Dataset, policy):

    qp.environ['SAMPLE_SIZE'] = 20

    base_quantifier=base_method(learner())

    if not dataset.binary and policy=='ds':
        print(f'skipping the test of binary policy ds on non-binary dataset {dataset}')
        return

    model = Ensemble(quantifier=base_quantifier, size=3, policy=policy, n_jobs=-1)

    model.fit(dataset.training)

    estim_prevalences = model.quantify(dataset.test.instances)

    true_prevalences = dataset.test.prevalence()
    error = qp.error.mae(true_prevalences, estim_prevalences)

    assert type(error) == np.float64


def test_quanet_method():
    try:
        import quapy.classification.neural
    except ModuleNotFoundError:
        print('skipping QuaNet test due to missing torch package')
        return

    qp.environ['SAMPLE_SIZE'] = 100

    # load the kindle dataset as text, and convert words to numerical indexes
    dataset = qp.datasets.fetch_reviews('kindle', pickle=True).reduce(200, 200)
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

    assert type(error) == np.float64


def test_str_label_names():
    model = qp.method.aggregative.CC(LogisticRegression())

    dataset = qp.datasets.fetch_reviews('imdb', pickle=True)
    dataset = Dataset(dataset.training.sampling(1000, *dataset.training.prevalence()),
                      dataset.test.sampling(1000, 0.25, 0.75))
    qp.data.preprocessing.text2tfidf(dataset, min_df=5, inplace=True)

    np.random.seed(0)
    model.fit(dataset.training)

    int_estim_prevalences = model.quantify(dataset.test.instances)
    true_prevalences = dataset.test.prevalence()

    error = qp.error.mae(true_prevalences, int_estim_prevalences)
    assert type(error) == np.float64

    dataset_str = Dataset(LabelledCollection(dataset.training.instances,
                                             ['one' if label == 1 else 'zero' for label in dataset.training.labels]),
                          LabelledCollection(dataset.test.instances,
                                             ['one' if label == 1 else 'zero' for label in dataset.test.labels]))
    assert all(dataset_str.training.classes_ == dataset_str.test.classes_), 'wrong indexation'
    np.random.seed(0)
    model.fit(dataset_str.training)

    str_estim_prevalences = model.quantify(dataset_str.test.instances)
    true_prevalences = dataset_str.test.prevalence()

    error = qp.error.mae(true_prevalences, str_estim_prevalences)
    assert type(error) == np.float64

    print(true_prevalences)
    print(int_estim_prevalences)
    print(str_estim_prevalences)

    np.testing.assert_almost_equal(int_estim_prevalences[1],
                                      str_estim_prevalences[list(model.classes_).index('one')])

# helper
def __fit_test(quantifier, train, test):
    quantifier.fit(train)
    test_samples = APP(test)
    true_prevs, estim_prevs = qp.evaluation.prediction(quantifier, test_samples)
    return qp.error.mae(true_prevs, estim_prevs), estim_prevs


def test_median_meta():
    """
    This test compares the performance of the MedianQuantifier with respect to computing the median of the predictions
    of a differently parameterized quantifier. We use the DistributionMatching base quantifier and the median is
    computed across different values of nbins
    """

    qp.environ['SAMPLE_SIZE'] = 100

    # grid of values
    nbins_grid = list(range(2, 11))

    dataset = 'kindle'
    train, test = qp.datasets.fetch_reviews(dataset, tfidf=True, min_df=10).train_test
    prevs = []
    errors = []
    for nbins in nbins_grid:
        with qp.util.temp_seed(0):
            q = DMy(LogisticRegression(), nbins=nbins)
            mae, estim_prevs = __fit_test(q, train, test)
            prevs.append(estim_prevs)
            errors.append(mae)
            print(f'{dataset} DistributionMatching(nbins={nbins}) got MAE {mae:.4f}')
    prevs = np.asarray(prevs)
    mae = np.mean(errors)
    print(f'\tMAE={mae:.4f}')

    q = DMy(LogisticRegression())
    q = MedianEstimator(q, param_grid={'nbins': nbins_grid}, random_state=0, n_jobs=-1)
    median_mae, prev = __fit_test(q, train, test)
    print(f'\tMAE={median_mae:.4f}')

    np.testing.assert_almost_equal(np.median(prevs, axis=0), prev)
    assert median_mae < mae, 'the median-based quantifier provided a higher error...'


def test_median_meta_modsel():
    """
    This test checks the median-meta quantifier with model selection
    """

    qp.environ['SAMPLE_SIZE'] = 100

    dataset = 'kindle'
    train, test = qp.datasets.fetch_reviews(dataset, tfidf=True, min_df=10).train_test
    train, val = train.split_stratified(random_state=0)

    nbins_grid = [2, 4, 5, 10, 15]

    q = DMy(LogisticRegression())
    q = MedianEstimator(q, param_grid={'nbins': nbins_grid}, random_state=0, n_jobs=-1)
    median_mae, _ = __fit_test(q, train, test)
    print(f'\tMAE={median_mae:.4f}')

    q = DMy(LogisticRegression())
    lr_params = {'classifier__C': np.logspace(-1, 1, 3)}
    q = MedianEstimator(q, param_grid={'nbins': nbins_grid}, random_state=0, n_jobs=-1)
    q = GridSearchQ(q, param_grid=lr_params, protocol=APP(val), n_jobs=-1)
    optimized_median_ave, _ = __fit_test(q, train, test)
    print(f'\tMAE={optimized_median_ave:.4f}')

    assert optimized_median_ave < median_mae, "the optimized method yielded worse performance..."