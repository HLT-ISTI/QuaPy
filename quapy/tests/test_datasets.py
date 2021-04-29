import pytest

from quapy.data.datasets import REVIEWS_SENTIMENT_DATASETS, TWITTER_SENTIMENT_DATASETS_TEST, \
    TWITTER_SENTIMENT_DATASETS_TRAIN, UCI_DATASETS, fetch_reviews, fetch_twitter, fetch_UCIDataset


@pytest.mark.parametrize('dataset_name', REVIEWS_SENTIMENT_DATASETS)
def test_fetch_reviews(dataset_name):
    fetch_reviews(dataset_name)


@pytest.mark.parametrize('dataset_name', TWITTER_SENTIMENT_DATASETS_TEST + TWITTER_SENTIMENT_DATASETS_TRAIN)
def test_fetch_twitter(dataset_name):
    fetch_twitter(dataset_name)

@pytest.mark.parametrize('dataset_name',  UCI_DATASETS)
def test_fetch_UCIDataset(dataset_name):
    fetch_UCIDataset(dataset_name)
