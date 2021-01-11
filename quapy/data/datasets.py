import zipfile
from util import download_file_if_not_exists, download_file, get_quapy_home, pickled_resource
import os
from os.path import join
from data.base import Dataset, LabelledCollection
from data.reader import *
from data.preprocessing import text2tfidf, reduce_columns
import pandas as pd


REVIEWS_SENTIMENT_DATASETS = ['hp', 'kindle', 'imdb']
TWITTER_SENTIMENT_DATASETS = ['gasp', 'hcr', 'omd', 'sanders',
                              'semeval13', 'semeval14', 'semeval15', 'semeval16',
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

    data.name = dataset_name

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

    data.name = dataset_name

    return data


UCI_DATASETS = ['acute.a', 'acute.b',
                'balance.1', 'balance.2', 'balance.3',
                'breast-cancer',
                'cmc.1', 'cmc.2', 'cmc.3',
                'ctg.1', 'ctg.2', 'ctg.3'] # ongoing...

def fetch_UCIDataset(dataset_name, data_home=None, verbose=False):

    assert dataset_name in UCI_DATASETS, \
        f'Name {dataset_name} does not match any known dataset from the UCI Machine Learning datasets repository. ' \
        f'Valid ones are {UCI_DATASETS}'
    if data_home is None:
        data_home = get_quapy_home()

    identifier_map = {
        'acute.a': 'acute',
        'acute.b': 'acute',
        'balance.1': 'balance-scale',
        'balance.2': 'balance-scale',
        'balance.3': 'balance-scale',
        'breast-cancer': 'breast-cancer-wisconsin',
        'cmc.1': 'cmc',
        'cmc.2': 'cmc',
        'cmc.3': 'cmc',
        'ctg.1': 'ctg',
        'ctg.2': 'ctg',
        'ctg.3': 'ctg',

    }

    dataset_fullname = {
        'acute.a': 'Acute Inflammations (urinary bladder)',
        'acute.b': 'Acute Inflammations (renal pelvis)',
        'balance.1': 'Balance Scale Weight & Distance Database (left)',
        'balance.2': 'Balance Scale Weight & Distance Database (balanced)',
        'balance.3': 'Balance Scale Weight & Distance Database (right)',
        'breast-cancer':  'Breast Cancer Wisconsin (Original)',
        'cmc.1': 'Contraceptive Method Choice (no use)',
        'cmc.2': 'Contraceptive Method Choice (long term)',
        'cmc.3': 'Contraceptive Method Choice (short term)',
        'ctg.1': 'Cardiotocography Data Set (normal)',
        'ctg.2': 'Cardiotocography Data Set (suspect)',
        'ctg.3': 'Cardiotocography Data Set (pathologic)',
    }

    data_folder = {
        'acute': 'diagnosis',
        'balance-scale': 'balance-scale',
        'breast-cancer-wisconsin': 'breast-cancer-wisconsin',
        'cmc': 'cmc'
    }

    identifier = identifier_map[dataset_name]
    URL = f'http://archive.ics.uci.edu/ml/machine-learning-databases/{identifier}'
    data_path = join(data_home, 'uci_datasets', identifier)
    download_file_if_not_exists(f'{URL}/{data_folder[identifier]}.data', f'{data_path}/{identifier}.data')
    download_file_if_not_exists(f'{URL}/{data_folder[identifier]}.names', f'{data_path}/{identifier}.names')

    if verbose:
        print(open(f'{data_path}/{identifier}.names', 'rt').read())

    print(f'Loading {dataset_name} ({dataset_fullname[dataset_name]})')
    if identifier == 'acute':
        df = pd.read_csv(f'{data_path}/{identifier}.data', header=None, encoding='utf-16', sep='\t')
        if dataset_name == 'acute.a':
            y = binarize(df[6], pos_class='yes')
        elif dataset_name == 'acute.b':
            y = binarize(df[7], pos_class='yes')

        mintemp, maxtemp = 35, 42
        df[0] = df[0].apply(lambda x:(float(x.replace(',','.'))-mintemp)/(maxtemp-mintemp)).astype(float, copy=False)
        [df_replace(df, col) for col in range(1, 6)]
        X = df.loc[:, 0:5].values

    if identifier == 'balance-scale':
        df = pd.read_csv(f'{data_path}/{identifier}.data', header=None, sep=',')
        if dataset_name == 'balance.1':
            y = binarize(df[0], pos_class='L')
        elif dataset_name == 'balance.2':
            y = binarize(df[0], pos_class='B')
        elif dataset_name == 'balance.3':
            y = binarize(df[0], pos_class='R')
        X = df.loc[:, 1:].astype(float).values

    if identifier == 'breast-cancer-wisconsin':
        df = pd.read_csv(f'{data_path}/{identifier}.data', header=None, sep=',')
        Xy = df.loc[:, 1:10]
        Xy[Xy=='?']=np.nan
        Xy = Xy.dropna(axis=0)
        X = Xy.loc[:, 1:9]
        X = X.astype(float).values
        y = binarize(Xy[10], pos_class=4)

    if identifier == 'cmc':
        df = pd.read_csv(f'{data_path}/{identifier}.data', header=None, sep=',')
        X = df.loc[:, 0:8].astype(float).values
        y = df[9].astype(int).values
        if dataset_name == 'cmc.1':
            y = binarize(y, pos_class=1)
        elif dataset_name == 'cmc.2':
            y = binarize(y, pos_class=2)
        elif dataset_name == 'cmc.3':
            y = binarize(y, pos_class=3)

    data = LabelledCollection(X, y)
    data.stats()
    raise NotImplementedError()
    #print(df)
    #print(df.loc[:, 0:5].values)
    #print(y)

#    X = __read_csv(f'{data_path}/{identifier}.data', separator='\t')
#    print(X)

    #X, y = from_csv(f'{data_path}/{dataset_name}.data')
    #y, classnames = reindex_labels(y)


#def __read_csv(path, separator=','):
#    x = []
#    for instance in tqdm(open(path, 'rt', encoding='utf-16').readlines(), desc=f'reading {path}'):
#        x.append(instance.strip().split(separator))
#    return x

def df_replace(df, col, repl={'yes': 1, 'no':0}, astype=float):
    df[col] = df[col].apply(lambda x:repl[x]).astype(astype, copy=False)