import zipfile
from util import download_file_if_not_exists, download_file, get_quapy_home, pickled_resource
import os
from os.path import join
from data.base import Dataset
from data.reader import from_text, from_sparse
from data.preprocessing import text2tfidf, reduce_columns


REVIEWS_SENTIMENT_DATASETS = ['hp', 'kindle', 'imdb']
TWITTER_SENTIMENT_DATASETS = ['gasp', 'hcr', 'omd', 'sanders', 'semeval13', 'semeval14', 'semeval15', 'semeval16',
                              'sst', 'wa', 'wb']


def fetch_reviews(dataset_name, tfidf=False, min_df=None, data_home=None, pickle=False):
    """
    Load a Reviews dataset as a Dataset instance, as used in:
    Esuli, A., Moreo, A., and Sebastiani, F. "A recurrent neural network for sentiment quantification."
    Proceedings of the 27th ACM International Conference on Information and Knowledge Management. 2018.
    :param dataset_name: the name of the dataset: valid ones are 'hp', 'kindle', 'imdb'
    :param tfidf: set to True to transform the raw documents into tfidf weighted matrices
    :param min_df: minimun number of documents that should contain a term in order for the term to be
    kept (ignored if tfidf==False)
    :param data_home: specify the quapy home directory where collections will be dumped (leave empty to use the default
    ~/quay_data/ directory)
    :param pickle: set to True to pickle the Dataset object the first time it is generated, in order to allow for
    faster subsequent invokations
    :return: a Dataset instance
    """
    assert dataset_name in REVIEWS_SENTIMENT_DATASETS, \
        f'Name {dataset_name} does not match any known dataset for sentiment reviews. ' \
        f'Valid ones are {REVIEWS_SENTIMENT_DATASETS}'
    if data_home is None:
        data_home = get_quapy_home()

    URL_TRAIN = f'https://zenodo.org/record/4117827/files/{dataset_name}_train.txt'
    URL_TEST = f'https://zenodo.org/record/4117827/files/{dataset_name}_test.txt'
    os.makedirs(join(data_home, 'reviews'), exist_ok=True)
    train_path = join(data_home, 'reviews', dataset_name, 'train.txt')
    test_path = join(data_home, 'reviews', dataset_name, 'test.txt')
    download_file_if_not_exists(URL_TRAIN, train_path)
    download_file_if_not_exists(URL_TEST, test_path)

    pickle_path = None
    if pickle:
        pickle_path = join(data_home, 'reviews', 'pickle', f'{dataset_name}.pkl')
    data = pickled_resource(pickle_path, Dataset.load, train_path, test_path, from_text)

    if tfidf:
        text2tfidf(data, inplace=True)
        if min_df is not None:
            reduce_columns(data, min_df=min_df, inplace=True)

    return data


def fetch_twitter(dataset_name, for_model_selection=False, min_df=None, data_home=None, pickle=False):
    """
    Load a Twitter dataset as a Dataset instance, as used in:
    Gao, W., Sebastiani, F.: From classification to quantification in tweet sentiment analysis.
    Social Network Analysis and Mining6(19), 1–22 (2016)

    :param dataset_name: the name of the dataset: valid ones are 'gasp', 'hcr', 'omd', 'sanders', 'semeval13',
    'semeval14', 'semeval15', 'semeval16', 'sst', 'wa', 'wb'
    :param for_model_selection: if True, then returns the train split as the training set and the devel split
    as the test set; if False, then returns the train+devel split as the training set and the test set as the
    test set
    :param min_df: minimun number of documents that should contain a term in order for the term to be kept
    :param data_home: specify the quapy home directory where collections will be dumped (leave empty to use the default
    ~/quay_data/ directory)
    :param pickle: set to True to pickle the Dataset object the first time it is generated, in order to allow for
    faster subsequent invokations
    :return: a Dataset instance
    """
    assert dataset_name in TWITTER_SENTIMENT_DATASETS, \
        f'Name {dataset_name} does not match any known dataset for sentiment twitter. ' \
        f'Valid ones are {TWITTER_SENTIMENT_DATASETS}'
    if data_home is None:
        data_home = get_quapy_home()

    URL = 'https://zenodo.org/record/4255764/files/tweet_sentiment_quantification_snam.zip'
    unzipped_path = join(data_home, 'tweet_sentiment_quantification_snam')
    if not os.path.exists(unzipped_path):
        downloaded_path = join(data_home, 'tweet_sentiment_quantification_snam.zip')
        download_file(URL, downloaded_path)
        with zipfile.ZipFile(downloaded_path) as file:
            file.extractall(data_home)
        os.remove(downloaded_path)

    if dataset_name in {'semeval13', 'semeval14', 'semeval15'}:
        trainset_name = 'semeval'
        testset_name  = 'semeval' if for_model_selection else dataset_name
        print(f"the training and development sets for datasets 'semeval13', 'semeval14', 'semeval15' are common "
              f"(called 'semeval'); returning trainin-set='{trainset_name}' and test-set={testset_name}")
    else:
        trainset_name = testset_name = dataset_name

    if for_model_selection:
        train = join(unzipped_path, 'train', f'{trainset_name}.train.feature.txt')
        test  = join(unzipped_path, 'test', f'{testset_name}.dev.feature.txt')
    else:
        train = join(unzipped_path, 'train', f'{trainset_name}.train+dev.feature.txt')
        if dataset_name == 'semeval16':  # there is a different test name in the case of semeval16 only
            test = join(unzipped_path, 'test', f'{testset_name}.dev-test.feature.txt')
        else:
            test = join(unzipped_path, 'test', f'{testset_name}.test.feature.txt')

    pickle_path = None
    if pickle:
        mode = "train-dev" if for_model_selection else "train+dev-test"
        pickle_path = join(unzipped_path, 'pickle', f'{testset_name}.{mode}.pkl')
    data = pickled_resource(pickle_path, Dataset.load, train, test, from_sparse)

    if min_df is not None:
        reduce_columns(data, min_df=min_df, inplace=True)

    return data



