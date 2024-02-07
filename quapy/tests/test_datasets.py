import pytest

from quapy.data.datasets import REVIEWS_SENTIMENT_DATASETS, TWITTER_SENTIMENT_DATASETS_TEST, \
    TWITTER_SENTIMENT_DATASETS_TRAIN, UCI_BINARY_DATASETS, LEQUA2022_TASKS, UCI_MULTICLASS_DATASETS,\
    fetch_reviews, fetch_twitter, fetch_UCIBinaryDataset, fetch_lequa2022, fetch_UCIMulticlassLabelledCollection


@pytest.mark.parametrize('dataset_name', REVIEWS_SENTIMENT_DATASETS)
def test_fetch_reviews(dataset_name):
    dataset = fetch_reviews(dataset_name)
    print(f'Dataset {dataset_name}')
    print('Training set stats')
    dataset.training.stats()
    print('Test set stats')
    dataset.test.stats()


@pytest.mark.parametrize('dataset_name', TWITTER_SENTIMENT_DATASETS_TEST + TWITTER_SENTIMENT_DATASETS_TRAIN)
def test_fetch_twitter(dataset_name):
    try:
        dataset = fetch_twitter(dataset_name)
    except ValueError as ve:
        if dataset_name == 'semeval' and ve.args[0].startswith(
                'dataset "semeval" can only be used for model selection.'):
            dataset = fetch_twitter(dataset_name, for_model_selection=True)
    print(f'Dataset {dataset_name}')
    print('Training set stats')
    dataset.training.stats()
    print('Test set stats')


@pytest.mark.parametrize('dataset_name', UCI_BINARY_DATASETS)
def test_fetch_UCIDataset(dataset_name):
    try:
        dataset = fetch_UCIBinaryDataset(dataset_name)
    except FileNotFoundError as fnfe:
        if dataset_name == 'pageblocks.5' and fnfe.args[0].find(
                'If this is the first time you attempt to load this dataset') > 0:
            print('The pageblocks.5 dataset requires some hand processing to be usable, skipping this test.')
            return
    print(f'Dataset {dataset_name}')
    print('Training set stats')
    dataset.training.stats()
    print('Test set stats')


@pytest.mark.parametrize('dataset_name', UCI_MULTICLASS_DATASETS)
def test_fetch_UCIMultiDataset(dataset_name):
    dataset = fetch_UCIMulticlassLabelledCollection(dataset_name)
    print(f'Dataset {dataset_name}')
    print('Training set stats')
    dataset.stats()
    print('Test set stats')


@pytest.mark.parametrize('dataset_name', LEQUA2022_TASKS)
def test_fetch_lequa2022(dataset_name):
    train, gen_val, gen_test = fetch_lequa2022(dataset_name)
    print(train.stats())
    print('Val:', gen_val.total())
    print('Test:', gen_test.total())
