import pytest

from quapy.data.datasets import REVIEWS_SENTIMENT_DATASETS, TWITTER_SENTIMENT_DATASETS_TEST, \
    TWITTER_SENTIMENT_DATASETS_TRAIN, UCI_DATASETS, fetch_reviews, fetch_twitter, fetch_UCIDataset


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


@pytest.mark.parametrize('dataset_name', UCI_DATASETS)
def test_fetch_UCIDataset(dataset_name):
    try:
        dataset = fetch_UCIDataset(dataset_name)
    except FileNotFoundError as fnfe:
        if dataset_name == 'pageblocks.5' and fnfe.args[0].find(
                'If this is the first time you attempt to load this dataset') > 0:
            print('The pageblocks.5 dataset requires some hand processing to be usable, skipping this test.')
            return
    print(f'Dataset {dataset_name}')
    print('Training set stats')
    dataset.training.stats()
    print('Test set stats')
