import pytest

from quapy.data.datasets import REVIEWS_SENTIMENT_DATASETS, TWITTER_SENTIMENT_DATASETS_TEST, \
    TWITTER_SENTIMENT_DATASETS_TRAIN, UCI_DATASETS, fetch_reviews, fetch_twitter, fetch_UCIDataset


@pytest.mark.parametrize('dataset_name', REVIEWS_SENTIMENT_DATASETS)
def test_fetch_reviews(dataset_name):
    dataset = fetch_reviews(dataset_name)
    print(dataset.n_classes, len(dataset.training), len(dataset.test))


@pytest.mark.parametrize('dataset_name', TWITTER_SENTIMENT_DATASETS_TEST + TWITTER_SENTIMENT_DATASETS_TRAIN)
def test_fetch_twitter(dataset_name):
    try:
        dataset = fetch_twitter(dataset_name)
    except ValueError as ve:
        if dataset_name == 'semeval' and ve.args[0].startswith(
                'dataset "semeval" can only be used for model selection.'):
            dataset = fetch_twitter(dataset_name, for_model_selection=True)
    print(dataset.n_classes, len(dataset.training), len(dataset.test))


@pytest.mark.parametrize('dataset_name', UCI_DATASETS)
def test_fetch_UCIDataset(dataset_name):
    try:
        dataset = fetch_UCIDataset(dataset_name)
    except FileNotFoundError as fnfe:
        if dataset_name == 'pageblocks.5' and fnfe.args[0].find(
                'If this is the first time you attempt to load this dataset') > 0:
            return
    print(dataset.n_classes, len(dataset.training), len(dataset.test))
