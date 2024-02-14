def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
import zipfile
from os.path import join
import pandas as pd
from ucimlrepo import fetch_ucirepo
from quapy.data.base import Dataset, LabelledCollection
from quapy.data.preprocessing import text2tfidf, reduce_columns
from quapy.data.reader import *
from quapy.util import download_file_if_not_exists, download_file, get_quapy_home, pickled_resource


REVIEWS_SENTIMENT_DATASETS = ['hp', 'kindle', 'imdb']
TWITTER_SENTIMENT_DATASETS_TEST = ['gasp', 'hcr', 'omd', 'sanders',
                              'semeval13', 'semeval14', 'semeval15', 'semeval16',
                              'sst', 'wa', 'wb']
TWITTER_SENTIMENT_DATASETS_TRAIN = ['gasp', 'hcr', 'omd', 'sanders',
                                 'semeval', 'semeval16',
                                 'sst', 'wa', 'wb']
UCI_BINARY_DATASETS = ['acute.a', 'acute.b',
                'balance.1', 'balance.2', 'balance.3',
                'breast-cancer',
                'cmc.1', 'cmc.2', 'cmc.3',
                'ctg.1', 'ctg.2', 'ctg.3',
                       #'diabetes', # <-- I haven't found this one...
                'german',
                'haberman',
                'ionosphere',
                'iris.1', 'iris.2', 'iris.3',
                'mammographic',
                       'pageblocks.5',
                       #'phoneme', # <-- I haven't found this one...
                       'semeion',
                       'sonar',
                       'spambase',
                       'spectf',
                       'tictactoe',
                       'transfusion',
                       'wdbc',
                       'wine.1', 'wine.2', 'wine.3',
                       'wine-q-red', 'wine-q-white',
                       'yeast']

UCI_MULTICLASS_DATASETS = ['dry-bean',
                           'wine-quality',
                           'academic-success',
                           'digits',
                           'letter']

LEQUA2022_TASKS = ['T1A', 'T1B', 'T2A', 'T2B']

_TXA_SAMPLE_SIZE = 250
_TXB_SAMPLE_SIZE = 1000

LEQUA2022_SAMPLE_SIZE = {
    'TXA': _TXA_SAMPLE_SIZE,
    'TXB': _TXB_SAMPLE_SIZE,
    'T1A': _TXA_SAMPLE_SIZE,
    'T1B': _TXB_SAMPLE_SIZE,
    'T2A': _TXA_SAMPLE_SIZE,
    'T2B': _TXB_SAMPLE_SIZE,
    'binary': _TXA_SAMPLE_SIZE,
    'multiclass': _TXB_SAMPLE_SIZE
}


def fetch_reviews(dataset_name, tfidf=False, min_df=None, data_home=None, pickle=False) -> Dataset:
    """
    Loads a Reviews dataset as a Dataset instance, as used in
    `Esuli, A., Moreo, A., and Sebastiani, F. "A recurrent neural network for sentiment quantification."
    Proceedings of the 27th ACM International Conference on Information and Knowledge Management. 2018. <https://dl.acm.org/doi/abs/10.1145/3269206.3269287>`_.
    The list of valid dataset names can be accessed in `quapy.data.datasets.REVIEWS_SENTIMENT_DATASETS`

    :param dataset_name: the name of the dataset: valid ones are 'hp', 'kindle', 'imdb'
    :param tfidf: set to True to transform the raw documents into tfidf weighted matrices
    :param min_df: minimun number of documents that should contain a term in order for the term to be
        kept (ignored if tfidf==False)
    :param data_home: specify the quapy home directory where collections will be dumped (leave empty to use the default
        ~/quay_data/ directory)
    :param pickle: set to True to pickle the Dataset object the first time it is generated, in order to allow for
        faster subsequent invokations
    :return: a :class:`quapy.data.base.Dataset` instance
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


def fetch_twitter(dataset_name, for_model_selection=False, min_df=None, data_home=None, pickle=False) -> Dataset:
    """
    Loads a Twitter dataset as a :class:`quapy.data.base.Dataset` instance, as used in:
    `Gao, W., Sebastiani, F.: From classification to quantification in tweet sentiment analysis.
    Social Network Analysis and Mining6(19), 1–22 (2016) <https://link.springer.com/content/pdf/10.1007/s13278-016-0327-z.pdf>`_
    Note that the datasets 'semeval13', 'semeval14', 'semeval15' share the same training set.
    The list of valid dataset names corresponding to training sets can be accessed in
    `quapy.data.datasets.TWITTER_SENTIMENT_DATASETS_TRAIN`, while the test sets can be accessed in
    `quapy.data.datasets.TWITTER_SENTIMENT_DATASETS_TEST`

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
    :return: a :class:`quapy.data.base.Dataset` instance
    """
    assert dataset_name in TWITTER_SENTIMENT_DATASETS_TRAIN + TWITTER_SENTIMENT_DATASETS_TEST, \
        f'Name {dataset_name} does not match any known dataset for sentiment twitter. ' \
        f'Valid ones are {TWITTER_SENTIMENT_DATASETS_TRAIN} for model selection and ' \
        f'{TWITTER_SENTIMENT_DATASETS_TEST} for test (datasets "semeval14", "semeval15", "semeval16" share ' \
        f'a common training set "semeval")'
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
        if dataset_name == 'semeval' and for_model_selection==False:
            raise ValueError('dataset "semeval" can only be used for model selection. '
                             'Use "semeval13", "semeval14", or "semeval15" for model evaluation.')
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


def fetch_UCIBinaryDataset(dataset_name, data_home=None, test_split=0.3, verbose=False) -> Dataset:
    """
    Loads a UCI dataset as an instance of :class:`quapy.data.base.Dataset`, as used in
    `Pérez-Gállego, P., Quevedo, J. R., & del Coz, J. J. (2017).
    Using ensembles for problems with characterizable changes in data distribution: A case study on quantification.
    Information Fusion, 34, 87-100. <https://www.sciencedirect.com/science/article/pii/S1566253516300628>`_
    and
    `Pérez-Gállego, P., Castano, A., Quevedo, J. R., & del Coz, J. J. (2019).
    Dynamic ensemble selection for quantification tasks.
    Information Fusion, 45, 1-15. <https://www.sciencedirect.com/science/article/pii/S1566253517303652>`_.
    The datasets do not come with a predefined train-test split (see :meth:`fetch_UCILabelledCollection` for further
    information on how to use these collections), and so a train-test split is generated at desired proportion.
    The list of valid dataset names can be accessed in `quapy.data.datasets.UCI_DATASETS`

    :param dataset_name: a dataset name
    :param data_home: specify the quapy home directory where collections will be dumped (leave empty to use the default
        ~/quay_data/ directory)
    :param test_split: proportion of documents to be included in the test set. The rest conforms the training set
    :param verbose: set to True (default is False) to get information (from the UCI ML repository) about the datasets
    :return: a :class:`quapy.data.base.Dataset` instance
    """
    data = fetch_UCIBinaryLabelledCollection(dataset_name, data_home, verbose)
    return Dataset(*data.split_stratified(1 - test_split, random_state=0))


def fetch_UCIBinaryLabelledCollection(dataset_name, data_home=None, verbose=False) -> LabelledCollection:
    """
    Loads a UCI collection as an instance of :class:`quapy.data.base.LabelledCollection`, as used in
    `Pérez-Gállego, P., Quevedo, J. R., & del Coz, J. J. (2017).
    Using ensembles for problems with characterizable changes in data distribution: A case study on quantification.
    Information Fusion, 34, 87-100. <https://www.sciencedirect.com/science/article/pii/S1566253516300628>`_
    and
    `Pérez-Gállego, P., Castano, A., Quevedo, J. R., & del Coz, J. J. (2019).
    Dynamic ensemble selection for quantification tasks.
    Information Fusion, 45, 1-15. <https://www.sciencedirect.com/science/article/pii/S1566253517303652>`_.
    The datasets do not come with a predefined train-test split, and so Pérez-Gállego et al. adopted a 5FCVx2 evaluation
    protocol, meaning that each collection was used to generate two rounds (hence the x2) of 5 fold cross validation.
    This can be reproduced by using :meth:`quapy.data.base.Dataset.kFCV`, e.g.:

    >>> import quapy as qp
    >>> collection = qp.datasets.fetch_UCIBinaryLabelledCollection("yeast")
    >>> for data in qp.train.Dataset.kFCV(collection, nfolds=5, nrepeats=2):
    >>>     ...

    The list of valid dataset names can be accessed in `quapy.data.datasets.UCI_DATASETS`

    :param dataset_name: a dataset name
    :param data_home: specify the quapy home directory where collections will be dumped (leave empty to use the default
        ~/quay_data/ directory)
    :param test_split: proportion of documents to be included in the test set. The rest conforms the training set
    :param verbose: set to True (default is False) to get information (from the UCI ML repository) about the datasets
    :return: a :class:`quapy.data.base.LabelledCollection` instance
    """

    assert dataset_name in UCI_BINARY_DATASETS, \
        f'Name {dataset_name} does not match any known dataset from the UCI Machine Learning datasets repository. ' \
        f'Valid ones are {UCI_BINARY_DATASETS}'
    if data_home is None:
        data_home = get_quapy_home()

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
        'german': 'Statlog German Credit Data',
        'haberman': "Haberman's Survival Data",
        'ionosphere': 'Johns Hopkins University Ionosphere DB',
        'iris.1': 'Iris Plants Database(x)',
        'iris.2': 'Iris Plants Database(versicolour)',
        'iris.3': 'Iris Plants Database(virginica)',
        'mammographic': 'Mammographic Mass',
        'pageblocks.5': 'Page Blocks Classification (5)',
        'semeion': 'Semeion Handwritten Digit (8)',
        'sonar': 'Sonar, Mines vs. Rocks',
        'spambase': 'Spambase Data Set',
        'spectf': 'SPECTF Heart Data',
        'tictactoe': 'Tic-Tac-Toe Endgame Database',
        'transfusion': 'Blood Transfusion Service Center Data Set',
        'wdbc': 'Wisconsin Diagnostic Breast Cancer',
        'wine.1': 'Wine Recognition Data (1)',
        'wine.2': 'Wine Recognition Data (2)',
        'wine.3': 'Wine Recognition Data (3)',
        'wine-q-red': 'Wine Quality Red (6-10)',
        'wine-q-white': 'Wine Quality White (6-10)',
        'yeast': 'Yeast',
    }

    # the identifier is an alias for the dataset group, it's part of the url data-folder, and is the name we use
    # to download the raw dataset
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
        'ctg.1': '00193',
        'ctg.2': '00193',
        'ctg.3': '00193',
        'german': 'statlog/german',
        'haberman': 'haberman',
        'ionosphere': 'ionosphere',
        'iris.1': 'iris',
        'iris.2': 'iris',
        'iris.3': 'iris',
        'mammographic': 'mammographic-masses',
        'pageblocks.5': 'page-blocks',
        'semeion': 'semeion',
        'sonar': 'undocumented/connectionist-bench/sonar',
        'spambase': 'spambase',
        'spectf': 'spect',
        'tictactoe': 'tic-tac-toe',
        'transfusion': 'blood-transfusion',
        'wdbc': 'breast-cancer-wisconsin',
        'wine-q-red': 'wine-quality',
        'wine-q-white': 'wine-quality',
        'wine.1': 'wine',
        'wine.2': 'wine',
        'wine.3': 'wine',
        'yeast': 'yeast',
    }

    # the filename is the name of the file within the data_folder indexed by the identifier
    file_name = {
        'acute': 'diagnosis.data',
        '00193': 'CTG.xls',
        'statlog/german': 'german.data-numeric',
        'mammographic-masses': 'mammographic_masses.data',
        'page-blocks': 'page-blocks.data.Z',
        'undocumented/connectionist-bench/sonar': 'sonar.all-data',
        'spect': ['SPECTF.train', 'SPECTF.test'],
        'blood-transfusion': 'transfusion.data',
        'wine-quality': ['winequality-red.csv', 'winequality-white.csv'],
        'breast-cancer-wisconsin': 'breast-cancer-wisconsin.data' if dataset_name=='breast-cancer' else 'wdbc.data'
    }

    # the filename containing the dataset description (if any)
    desc_name = {
        'acute': 'diagnosis.names',
        '00193': None,
        'statlog/german': 'german.doc',
        'mammographic-masses': 'mammographic_masses.names',
        'undocumented/connectionist-bench/sonar': 'sonar.names',
        'spect': 'SPECTF.names',
        'blood-transfusion': 'transfusion.names',
        'wine-quality': 'winequality.names',
        'breast-cancer-wisconsin': 'breast-cancer-wisconsin.names' if dataset_name == 'breast-cancer' else 'wdbc.names'
    }

    identifier = identifier_map[dataset_name]
    filename = file_name.get(identifier, f'{identifier}.data')
    descfile = desc_name.get(identifier, f'{identifier}.names')
    fullname = dataset_fullname[dataset_name]

    URL = f'http://archive.ics.uci.edu/ml/machine-learning-databases/{identifier}'
    data_dir = join(data_home, 'uci_datasets', identifier)
    if isinstance(filename, str):  # filename could be a list of files, in which case it will be processed later
        data_path = join(data_dir, filename)
        download_file_if_not_exists(f'{URL}/{filename}', data_path)

    if descfile:
        try:
            download_file_if_not_exists(f'{URL}/{descfile}', f'{data_dir}/{descfile}')
            if verbose:
                print(open(f'{data_dir}/{descfile}', 'rt').read())
        except Exception:
            print('could not read the description file')
    elif verbose:
        print('no file description available')

    if verbose:
        print(f'Loading {dataset_name} ({fullname})')
    if identifier == 'acute':
        df = pd.read_csv(data_path, header=None, encoding='utf-16', sep='\t')

        df[0] = df[0].apply(lambda x: float(x.replace(',', '.'))).astype(float, copy=False)
        [_df_replace(df, col) for col in range(1, 6)]
        X = df.loc[:, 0:5].values
        if dataset_name == 'acute.a':
            y = binarize(df[6], pos_class='yes')
        elif dataset_name == 'acute.b':
            y = binarize(df[7], pos_class='yes')

    if identifier == 'balance-scale':
        df = pd.read_csv(data_path, header=None, sep=',')
        if dataset_name == 'balance.1':
            y = binarize(df[0], pos_class='L')
        elif dataset_name == 'balance.2':
            y = binarize(df[0], pos_class='B')
        elif dataset_name == 'balance.3':
            y = binarize(df[0], pos_class='R')
        X = df.loc[:, 1:].astype(float).values

    if identifier == 'breast-cancer-wisconsin' and dataset_name=='breast-cancer':
        df = pd.read_csv(data_path, header=None, sep=',')
        Xy = df.loc[:, 1:10]
        Xy[Xy=='?']=np.nan
        Xy = Xy.dropna(axis=0)
        X = Xy.loc[:, 1:9]
        X = X.astype(float).values
        y = binarize(Xy[10], pos_class=2)

    if identifier == 'breast-cancer-wisconsin' and dataset_name=='wdbc':
        df = pd.read_csv(data_path, header=None, sep=',')
        X = df.loc[:, 2:32].astype(float).values
        y = df[1].values
        y = binarize(y, pos_class='M')

    if identifier == 'cmc':
        df = pd.read_csv(data_path, header=None, sep=',')
        X = df.loc[:, 0:8].astype(float).values
        y = df[9].astype(int).values
        if dataset_name == 'cmc.1':
            y = binarize(y, pos_class=1)
        elif dataset_name == 'cmc.2':
            y = binarize(y, pos_class=2)
        elif dataset_name == 'cmc.3':
            y = binarize(y, pos_class=3)

    if identifier == '00193':
        df = pd.read_excel(data_path, sheet_name='Data', skipfooter=3)
        df = df[list(range(1,24))] # select columns numbered (number 23 is the target label)
        # replaces the header with the first row
        new_header = df.iloc[0]  # grab the first row for the header
        df = df[1:]  # take the data less the header row
        df.columns = new_header  # set the header row as the df header
        X = df.iloc[:, 0:22].astype(float).values
        y = df['NSP'].astype(int).values
        if dataset_name == 'ctg.1':
            y = binarize(y, pos_class=1)  # 1==Normal
        elif dataset_name == 'ctg.2':
            y = binarize(y, pos_class=2)  # 2==Suspect
        elif dataset_name == 'ctg.3':
            y = binarize(y, pos_class=3)  # 3==Pathologic

    if identifier == 'statlog/german':
        df = pd.read_csv(data_path, header=None, delim_whitespace=True)
        X = df.iloc[:, 0:24].astype(float).values
        y = df[24].astype(int).values
        y = binarize(y, pos_class=1)

    if identifier == 'haberman':
        df = pd.read_csv(data_path, header=None)
        X = df.iloc[:, 0:3].astype(float).values
        y = df[3].astype(int).values
        y = binarize(y, pos_class=2)

    if identifier == 'ionosphere':
        df = pd.read_csv(data_path, header=None)
        X = df.iloc[:, 0:34].astype(float).values
        y = df[34].values
        y = binarize(y, pos_class='b')

    if identifier == 'iris':
        df = pd.read_csv(data_path, header=None)
        X = df.iloc[:, 0:4].astype(float).values
        y = df[4].values
        if dataset_name == 'iris.1':
            y = binarize(y, pos_class='Iris-setosa')  # 1==Setosa
        elif dataset_name == 'iris.2':
            y = binarize(y, pos_class='Iris-versicolor')  # 2==Versicolor
        elif dataset_name == 'iris.3':
            y = binarize(y, pos_class='Iris-virginica')  # 3==Virginica

    if identifier == 'mammographic-masses':
        df = pd.read_csv(data_path, header=None, sep=',')
        df[df == '?'] = np.nan
        Xy = df.dropna(axis=0)
        X = Xy.iloc[:, 0:5]
        X = X.astype(float).values
        y = binarize(Xy.iloc[:,5], pos_class=1)

    if identifier == 'page-blocks':
        data_path_ = data_path.replace('.Z', '')
        if not os.path.exists(data_path_):
            raise FileNotFoundError(f'Warning: file {data_path_} does not exist. If this is the first time you '
                                    f'attempt to load this dataset, then you have to manually unzip the {data_path} '
                                    f'and name the extracted file {data_path_} (unfortunately, neither zipfile, nor '
                                    f'gzip can handle unix compressed files automatically -- there is a repo in GitHub '
                                    f'https://github.com/umeat/unlzw where the problem seems to be solved anyway).')
        df = pd.read_csv(data_path_, header=None, delim_whitespace=True)
        X = df.iloc[:, 0:10].astype(float).values
        y = df[10].values
        y = binarize(y, pos_class=5)  # 5==block "graphic"

    if identifier == 'semeion':
        df = pd.read_csv(data_path, header=None, delim_whitespace=True )
        X = df.iloc[:, 0:256].astype(float).values
        y = df[263].values  # 263 stands for digit 8 (labels are one-hot vectors from col 256-266)
        y = binarize(y, pos_class=1)

    if identifier == 'undocumented/connectionist-bench/sonar':
        df = pd.read_csv(data_path, header=None, sep=',')
        X = df.iloc[:, 0:60].astype(float).values
        y = df[60].values
        y = binarize(y, pos_class='R')

    if identifier == 'spambase':
        df = pd.read_csv(data_path, header=None, sep=',')
        X = df.iloc[:, 0:57].astype(float).values
        y = df[57].values
        y = binarize(y, pos_class=1)

    if identifier == 'spect':
        dfs = []
        for file in filename:
            data_path = join(data_dir, file)
            download_file_if_not_exists(f'{URL}/{file}', data_path)
            dfs.append(pd.read_csv(data_path, header=None, sep=','))
        df = pd.concat(dfs)
        X = df.iloc[:, 1:45].astype(float).values
        y = df[0].values
        y = binarize(y, pos_class=0)

    if identifier == 'tic-tac-toe':
        df = pd.read_csv(data_path, header=None, sep=',')
        X = df.iloc[:, 0:9].replace('o',0).replace('b',1).replace('x',2).values
        y = df[9].values
        y = binarize(y, pos_class='negative')

    if identifier == 'blood-transfusion':
        df = pd.read_csv(data_path, sep=',')
        X = df.iloc[:, 0:4].astype(float).values
        y = df.iloc[:, 4].values
        y = binarize(y, pos_class=1)

    if identifier == 'wine':
        df = pd.read_csv(data_path, header=None, sep=',')
        X = df.iloc[:, 1:14].astype(float).values
        y = df[0].values
        if dataset_name == 'wine.1':
            y = binarize(y, pos_class=1)
        elif dataset_name == 'wine.2':
            y = binarize(y, pos_class=2)
        elif dataset_name == 'wine.3':
            y = binarize(y, pos_class=3)

    if identifier == 'wine-quality':
        filename = filename[0] if dataset_name=='wine-q-red' else filename[1]
        data_path = join(data_dir, filename)
        download_file_if_not_exists(f'{URL}/{filename}', data_path)
        df = pd.read_csv(data_path, sep=';')
        X = df.iloc[:, 0:11].astype(float).values
        y = df.iloc[:, 11].values > 5

    if identifier == 'yeast':
        df = pd.read_csv(data_path, header=None, delim_whitespace=True)
        X = df.iloc[:, 1:9].astype(float).values
        y = df.iloc[:, 9].values
        y = binarize(y, pos_class='NUC')

    data = LabelledCollection(X, y)
    if verbose:
        data.stats()
    return data


def fetch_UCIMulticlassDataset(dataset_name, data_home=None, test_split=0.3, verbose=False) -> Dataset:
    """
    Loads a UCI multiclass dataset as an instance of :class:`quapy.data.base.Dataset`. 

    The list of available datasets is taken from https://archive.ics.uci.edu/, following these criteria:
    - It has more than 1000 instances
    - It is suited for classification
    - It has more than two classes
    - It is available for Python import (requires ucimlrepo package)

    >>> import quapy as qp
    >>> dataset = qp.datasets.fetch_UCIMulticlassDataset("dry-bean")
    >>> train, test = dataset.train_test
    >>>     ...

    The list of valid dataset names can be accessed in `quapy.data.datasets.UCI_MULTICLASS_DATASETS`

    The datasets are downloaded only once and pickled into disk, saving time for consecutive calls.

    :param dataset_name: a dataset name
    :param data_home: specify the quapy home directory where collections will be dumped (leave empty to use the default
        ~/quay_data/ directory)
    :param test_split: proportion of documents to be included in the test set. The rest conforms the training set
    :param verbose: set to True (default is False) to get information (stats) about the dataset
    :return: a :class:`quapy.data.base.Dataset` instance
    """
    data = fetch_UCIMulticlassLabelledCollection(dataset_name, data_home, verbose)
    return Dataset(*data.split_stratified(1 - test_split, random_state=0))


def fetch_UCIMulticlassLabelledCollection(dataset_name, data_home=None, verbose=False) -> LabelledCollection:
    """
    Loads a UCI multiclass collection as an instance of :class:`quapy.data.base.LabelledCollection`.

    The list of available datasets is taken from https://archive.ics.uci.edu/, following these criteria:
    - It has more than 1000 instances
    - It is suited for classification
    - It has more than two classes
    - It is available for Python import (requires ucimlrepo package)
    
    >>> import quapy as qp
    >>> collection = qp.datasets.fetch_UCIMulticlassLabelledCollection("dry-bean")
    >>> X, y = collection.Xy
    >>>     ...

    The list of valid dataset names can be accessed in `quapy.data.datasets.UCI_MULTICLASS_DATASETS`

    The datasets are downloaded only once and pickled into disk, saving time for consecutive calls.

    :param dataset_name: a dataset name
    :param data_home: specify the quapy home directory where the dataset will be dumped (leave empty to use the default
        ~/quay_data/ directory)
    :param test_split: proportion of documents to be included in the test set. The rest conforms the training set
    :param verbose: set to True (default is False) to get information (stats) about the dataset
    :return: a :class:`quapy.data.base.LabelledCollection` instance
    """
    assert dataset_name in UCI_MULTICLASS_DATASETS, \
        f'Name {dataset_name} does not match any known dataset from the ' \
        f'UCI Machine Learning datasets repository (multiclass). ' \
        f'Valid ones are {UCI_MULTICLASS_DATASETS}'
    
    if data_home is None:
        data_home = get_quapy_home()
    
    identifiers = {
        "dry-bean": 602,
        "wine-quality": 186,
        "academic-success": 697,
        "digits": 80,
        "letter": 59
    }
    
    full_names = {
        "dry-bean": "Dry Bean Dataset",
        "wine-quality": "Wine Quality",
        "academic-success": "Predict students' dropout and academic success",
        "digits": "Optical Recognition of Handwritten Digits",
        "letter": "Letter Recognition"
    }
    
    identifier = identifiers[dataset_name]
    fullname = full_names[dataset_name]

    if verbose:
        print(f'Loading UCI Muticlass {dataset_name} ({fullname})')

    file = join(data_home, 'uci_multiclass', dataset_name+'.pkl')
    
    def download(id):
        data = fetch_ucirepo(id=id)
        X, y = data['data']['features'].to_numpy(), data['data']['targets'].to_numpy().squeeze()
        classes = np.sort(np.unique(y))
        y = np.searchsorted(classes, y)
        return LabelledCollection(X, y)

    data = pickled_resource(file, download, identifier)

    if verbose:
        data.stats()
        
    return data


def _df_replace(df, col, repl={'yes': 1, 'no':0}, astype=float):
    df[col] = df[col].apply(lambda x:repl[x]).astype(astype, copy=False)


def fetch_lequa2022(task, data_home=None):
    """
    Loads the official datasets provided for the `LeQua <https://lequa2022.github.io/index>`_ competition.
    In brief, there are 4 tasks (T1A, T1B, T2A, T2B) having to do with text quantification
    problems. Tasks T1A and T1B provide documents in vector form, while T2A and T2B provide raw documents instead.
    Tasks T1A and T2A are binary sentiment quantification problems, while T2A and T2B are multiclass quantification
    problems consisting of estimating the class prevalence values of 28 different merchandise products.
    We refer to the `Esuli, A., Moreo, A., Sebastiani, F., & Sperduti, G. (2022).
    A Detailed Overview of LeQua@ CLEF 2022: Learning to Quantify.
    <https://ceur-ws.org/Vol-3180/paper-146.pdf>`_ for a detailed description
    on the tasks and datasets.

    The datasets are downloaded only once, and stored for fast reuse.

    See `lequa2022_experiments.py` provided in the example folder, that can serve as a guide on how to use these
    datasets.


    :param task: a string representing the task name; valid ones are T1A, T1B, T2A, and T2B
    :param data_home: specify the quapy home directory where collections will be dumped (leave empty to use the default
        ~/quay_data/ directory)
    :return: a tuple `(train, val_gen, test_gen)` where `train` is an instance of
        :class:`quapy.data.base.LabelledCollection`, `val_gen` and `test_gen` are instances of
        :class:`quapy.data._lequa2022.SamplesFromDir`, a subclass of :class:`quapy.protocol.AbstractProtocol`,
        that return a series of samples stored in a directory which are labelled by prevalence.
    """

    from quapy.data._lequa2022 import load_raw_documents, load_vector_documents, SamplesFromDir

    assert task in LEQUA2022_TASKS, \
        f'Unknown task {task}. Valid ones are {LEQUA2022_TASKS}'
    if data_home is None:
        data_home = get_quapy_home()

    URL_TRAINDEV=f'https://zenodo.org/record/6546188/files/{task}.train_dev.zip'
    URL_TEST=f'https://zenodo.org/record/6546188/files/{task}.test.zip'
    URL_TEST_PREV=f'https://zenodo.org/record/6546188/files/{task}.test_prevalences.zip'

    lequa_dir = join(data_home, 'lequa2022')
    os.makedirs(lequa_dir, exist_ok=True)

    def download_unzip_and_remove(unzipped_path, url):
        tmp_path = join(lequa_dir, task + '_tmp.zip')
        download_file_if_not_exists(url, tmp_path)
        with zipfile.ZipFile(tmp_path) as file:
            file.extractall(unzipped_path)
        os.remove(tmp_path)

    if not os.path.exists(join(lequa_dir, task)):
        download_unzip_and_remove(lequa_dir, URL_TRAINDEV)
        download_unzip_and_remove(lequa_dir, URL_TEST)
        download_unzip_and_remove(lequa_dir, URL_TEST_PREV)

    if task in ['T1A', 'T1B']:
        load_fn = load_vector_documents
    elif task in ['T2A', 'T2B']:
        load_fn = load_raw_documents

    tr_path = join(lequa_dir, task, 'public', 'training_data.txt')
    train = LabelledCollection.load(tr_path, loader_func=load_fn)

    val_samples_path = join(lequa_dir, task, 'public', 'dev_samples')
    val_true_prev_path = join(lequa_dir, task, 'public', 'dev_prevalences.txt')
    val_gen = SamplesFromDir(val_samples_path, val_true_prev_path, load_fn=load_fn)

    test_samples_path = join(lequa_dir, task, 'public', 'test_samples')
    test_true_prev_path = join(lequa_dir, task, 'public', 'test_prevalences.txt')
    test_gen = SamplesFromDir(test_samples_path, test_true_prev_path, load_fn=load_fn)

    return train, val_gen, test_gen


def fetch_IFCB(single_sample_train=True, for_model_selection=False, data_home=None):
    """
    Loads the IFCB dataset for quantification from `Zenodo <https://zenodo.org/records/10036244>`_ (for more
    information on this dataset, please follow the zenodo link).
    This dataset is based on the data available publicly at
    `WHOI-Plankton repo <https://github.com/hsosik/WHOI-Plankton>`_.
    The scripts for the processing are available at `P. González's repo <https://github.com/pglez82/IFCB_Zenodo>`_.
    Basically, this is the IFCB dataset with precomputed features for testing quantification algorithms.

    The datasets are downloaded only once, and stored for fast reuse.

    :param single_sample_train: a boolean. If true, it will return the train dataset as a
        :class:`quapy.data.base.LabelledCollection` (all examples together).
        If false, a generator of training samples will be returned. Each example in the training set has an individual label.
    :param for_model_selection: if True, then returns a split 30% of the training set (86 out of 286 samples) to be used for model selection; 
        if False, then returns the full training set as training set and the test set as the test set
    :param data_home: specify the quapy home directory where collections will be dumped (leave empty to use the default
        ~/quay_data/ directory)
    :return: a tuple `(train, test_gen)` where `train` is an instance of
        :class:`quapy.data.base.LabelledCollection`, if `single_sample_train` is true or
        :class:`quapy.data._ifcb.IFCBTrainSamplesFromDir`, i.e. a sampling protocol that returns a series of samples
        labelled example by example. test_gen will be a :class:`quapy.data._ifcb.IFCBTestSamples`, 
        i.e., a sampling protocol that returns a series of samples labelled by prevalence.
    """

    from quapy.data._ifcb import IFCBTrainSamplesFromDir, IFCBTestSamples, get_sample_list, generate_modelselection_split

    if data_home is None:
        data_home = get_quapy_home()
    
    URL_TRAIN=f'https://zenodo.org/records/10036244/files/IFCB.train.zip'
    URL_TEST=f'https://zenodo.org/records/10036244/files/IFCB.test.zip'
    URL_TEST_PREV=f'https://zenodo.org/records/10036244/files/IFCB.test_prevalences.zip'

    ifcb_dir = join(data_home, 'ifcb')
    os.makedirs(ifcb_dir, exist_ok=True)

    def download_unzip_and_remove(unzipped_path, url):
        tmp_path = join(ifcb_dir, 'ifcb_tmp.zip')
        download_file_if_not_exists(url, tmp_path)
        with zipfile.ZipFile(tmp_path) as file:
            file.extractall(unzipped_path)
        os.remove(tmp_path)

    if not os.path.exists(os.path.join(ifcb_dir,'train')):
        download_unzip_and_remove(ifcb_dir, URL_TRAIN)
    if not os.path.exists(os.path.join(ifcb_dir,'test')):
        download_unzip_and_remove(ifcb_dir, URL_TEST)
    if not os.path.exists(os.path.join(ifcb_dir,'test_prevalences.csv')):
        download_unzip_and_remove(ifcb_dir, URL_TEST_PREV)

    # Load test prevalences and classes
    test_true_prev_path = join(ifcb_dir, 'test_prevalences.csv')
    test_true_prev = pd.read_csv(test_true_prev_path)
    classes = test_true_prev.columns[1:]

    #Load train and test samples
    train_samples_path = join(ifcb_dir,'train')
    test_samples_path = join(ifcb_dir,'test')

    if for_model_selection:
        # In this case, return 70% of training data as the training set and 30% as the test set
        samples = get_sample_list(train_samples_path)
        train, test = generate_modelselection_split(samples, split=0.3)
        train_gen = IFCBTrainSamplesFromDir(path_dir=train_samples_path, classes=classes, samples=train)

        # Test prevalence is computed from class labels
        test_gen = IFCBTestSamples(path_dir=train_samples_path, test_prevalences=None, samples=test, classes=classes)
    else:
        # In this case, we use all training samples as the training set and the test samples as the test set
        train_gen = IFCBTrainSamplesFromDir(path_dir=train_samples_path, classes=classes)
        test_gen = IFCBTestSamples(path_dir=test_samples_path, test_prevalences=test_true_prev)

    # In the case the user wants it, join all the train samples in one LabelledCollection
    if single_sample_train:
        train = LabelledCollection.join(*[lc for lc in train_gen()])
        return train, test_gen
    else:
        return train_gen, test_gen
