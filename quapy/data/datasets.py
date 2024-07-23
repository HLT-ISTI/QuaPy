def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
from contextlib import contextmanager
import zipfile
from os.path import join
import pandas as pd
from ucimlrepo import fetch_ucirepo
from quapy.data.base import Dataset, LabelledCollection
from quapy.data.preprocessing import text2tfidf, reduce_columns
from quapy.data.preprocessing import standardize as standardizer
from quapy.data.reader import *
from quapy.util import download_file_if_not_exists, download_file, get_quapy_home, pickled_resource
from sklearn.preprocessing import StandardScaler


REVIEWS_SENTIMENT_DATASETS = ['hp', 'kindle', 'imdb']

TWITTER_SENTIMENT_DATASETS_TEST = [
    'gasp', 'hcr', 'omd', 'sanders',
    'semeval13', 'semeval14', 'semeval15', 'semeval16',
    'sst', 'wa', 'wb',
]

TWITTER_SENTIMENT_DATASETS_TRAIN = [
    'gasp', 'hcr', 'omd', 'sanders',
    'semeval', 'semeval16',
    'sst', 'wa', 'wb',
]

UCI_BINARY_DATASETS = [
    #'acute.a', 'acute.b',
    'balance.1', 
    #'balance.2', 
    'balance.3',
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
    'wine-q-red',
    'wine-q-white',
    'yeast',
]

UCI_MULTICLASS_DATASETS = [
    'dry-bean',
    'wine-quality',
    'academic-success',
    'digits',
    'letter',
    'abalone',
    'obesity',
    'nursery',
    'yeast',
    'hand_digits',
    'satellite',
    'shuttle',
    'cmc',
    'isolet',
    'waveform-v1',
    'molecular',
    'poker_hand',
    'connect-4',
    'mhr',
    'chess',
    'page_block',
    'phishing',
    'image_seg',
    'hcv',
]

LEQUA2022_VECTOR_TASKS = ['T1A', 'T1B']
LEQUA2022_TEXT_TASKS = ['T2A', 'T2B']
LEQUA2022_TASKS = LEQUA2022_VECTOR_TASKS + LEQUA2022_TEXT_TASKS

LEQUA2024_TASKS = ['T1', 'T2', 'T3', 'T4']

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

LEQUA2024_SAMPLE_SIZE = {
    'T1': 250,
    'T2': 1000,
    'T3': 200,
    'T4': 250,
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


def fetch_UCIBinaryDataset(dataset_name, data_home=None, test_split=0.3, standardize=True, verbose=False) -> Dataset:
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
    :param standardize: indicates whether the covariates should be standardized or not (default is True). If requested,
        standardization applies after the LabelledCollection is split, that is, the mean an std are computed only on the
        training portion of the data.
    :param verbose: set to True (default is False) to get information (from the UCI ML repository) about the datasets
    :return: a :class:`quapy.data.base.Dataset` instance
    """
    data = fetch_UCIBinaryLabelledCollection(dataset_name, data_home, verbose)
    dataset = Dataset(*data.split_stratified(1 - test_split, random_state=0), name=dataset_name)
    if standardize:
        dataset = standardizer(dataset)
    return dataset


def fetch_UCIBinaryLabelledCollection(dataset_name, data_home=None, standardize=True, verbose=False) -> LabelledCollection:
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
    >>> for data in qp.datasets.Dataset.kFCV(collection, nfolds=5, nrepeats=2):
    >>>     ...

    The list of valid dataset names can be accessed in `quapy.data.datasets.UCI_DATASETS`

    :param dataset_name: a dataset name
    :param data_home: specify the quapy home directory where collections will be dumped (leave empty to use the default
        ~/quay_data/ directory)
    :param standardize: indicates whether the covariates should be standardized or not (default is True). 
    :param verbose: set to True (default is False) to get information (from the UCI ML repository) about the datasets
    :return: a :class:`quapy.data.base.LabelledCollection` instance
    """
    assert dataset_name in UCI_BINARY_DATASETS, (
        f"Name {dataset_name} does not match any known dataset from the UCI Machine Learning datasets repository. "
        f"Valid ones are {UCI_BINARY_DATASETS}"
    )
    if data_home is None:
        data_home = get_quapy_home()

    # mapping bewteen dataset names and UCI api ids
    identifiers = {
        "acute.a": 184,
        "acute.b": 184,
        "balance.1": 12,
        "balance.2": 12,
        "balance.3": 12,
        "breast-cancer": 15,
        "cmc.1": 30,
        "cmc.2": 30,
        "cmc.3": 30,
        # "ctg.1": ,  # not python importable
        # "ctg.2": ,  # not python importable
        # "ctg.3": ,  # not python importable
        # "german": ,  # not python importable
        "haberman": 43,
        "ionosphere": 52,
        "iris.1": 53,
        "iris.2": 53,
        "iris.3": 53,
        "mammographic": 161,
        "pageblocks.5": 78,
        # "semeion": ,  # not python importable
        "sonar": 151,
        "spambase": 94,
        "spectf": 96,
        "tictactoe": 101,
        "transfusion": 176,
        "wdbc": 17,
        "wine.1": 109,
        "wine.2": 109,
        "wine.3": 109,
        "wine-q-red": 186,
        "wine-q-white": 186,
        "yeast": 110,
    }

    # mapping between dataset names and dataset groups
    groups = {
        "acute.a": "acute",
        "acute.b": "acute",
        "balance.1": "balance",
        "balance.2": "balance",
        "balance.3": "balance",
        "breast-cancer": "breast-cancer",
        "cmc.1": "cmc",
        "cmc.2": "cmc",
        "cmc.3": "cmc",
        "ctg.1": "ctg",
        "ctg.2": "ctg",
        "ctg.3": "ctg",
        "german": "german",
        "haberman": "haberman",
        "ionosphere": "ionosphere",
        "iris.1": "iris",
        "iris.2": "iris",
        "iris.3": "iris",
        "mammographic": "mammographic",
        "pageblocks.5": "pageblocks",
        "semeion": "semeion",
        "sonar": "sonar",
        "spambase": "spambase",
        "spectf": "spectf",
        "tictactoe": "tictactoe",
        "transfusion": "transfusion",
        "wdbc": "wdbc",
        "wine-q-red": "wine-quality",
        "wine-q-white": "wine-quality",
        "wine.1": "wine",
        "wine.2": "wine",
        "wine.3": "wine",
        "yeast": "yeast",
    }

    # mapping between dataset short names and full names
    full_names = {
        "acute.a": "Acute Inflammations (urinary bladder)",
        "acute.b": "Acute Inflammations (renal pelvis)",
        "balance.1": "Balance Scale Weight & Distance Database (left)",
        "balance.2": "Balance Scale Weight & Distance Database (balanced)",
        "balance.3": "Balance Scale Weight & Distance Database (right)",
        "breast-cancer": "Breast Cancer Wisconsin (Original)",
        "cmc.1": "Contraceptive Method Choice (no use)",
        "cmc.2": "Contraceptive Method Choice (long term)",
        "cmc.3": "Contraceptive Method Choice (short term)",
        "ctg.1": "Cardiotocography Data Set (normal)",
        "ctg.2": "Cardiotocography Data Set (suspect)",
        "ctg.3": "Cardiotocography Data Set (pathologic)",
        "german": "Statlog German Credit Data",
        "haberman": "Haberman's Survival Data",
        "ionosphere": "Johns Hopkins University Ionosphere DB",
        "iris.1": "Iris Plants Database(x)",
        "iris.2": "Iris Plants Database(versicolour)",
        "iris.3": "Iris Plants Database(virginica)",
        "mammographic": "Mammographic Mass",
        "pageblocks.5": "Page Blocks Classification (5)",
        "semeion": "Semeion Handwritten Digit (8)",
        "sonar": "Sonar, Mines vs. Rocks",
        "spambase": "Spambase Data Set",
        "spectf": "SPECTF Heart Data",
        "tictactoe": "Tic-Tac-Toe Endgame Database",
        "transfusion": "Blood Transfusion Service Center Data Set",
        "wdbc": "Wisconsin Diagnostic Breast Cancer",
        "wine.1": "Wine Recognition Data (1)",
        "wine.2": "Wine Recognition Data (2)",
        "wine.3": "Wine Recognition Data (3)",
        "wine-q-red": "Wine Quality Red (6-10)",
        "wine-q-white": "Wine Quality White (6-10)",
        "yeast": "Yeast",
    }

    # mapping between dataset names and values of positive class
    pos_class = {
        "acute.a": "yes",
        "acute.b": "yes",
        "balance.1": "L",
        "balance.2": "B",
        "balance.3": "R",
        "breast-cancer": 2,
        "cmc.1": 1,
        "cmc.2": 2,
        "cmc.3": 3,
        "ctg.1": 1,  # 1==Normal
        "ctg.2": 2,  # 2==Suspect
        "ctg.3": 3,  # 3==Pathologic
        "german": 1,
        "haberman": 2,
        "ionosphere": "b",
        "iris.1": "Iris-setosa",  # 1==Setosa
        "iris.2": "Iris-versicolor",  # 2==Versicolor
        "iris.3": "Iris-virginica",  # 3==Virginica
        "mammographic": 1,
        "pageblocks.5": 5,  # 5==block "graphic"
        "semeion": 1,
        "sonar": "R",
        "spambase": 1,
        "spectf": 0,
        "tictactoe": "negative",
        "transfusion": 1,
        "wdbc": "M",
        "wine.1": 1,
        "wine.2": 2,
        "wine.3": 3,
        "wine-q-red": 1,
        "wine-q-white": 1,
        "yeast": "NUC",
    }

    identifier = identifiers.get(dataset_name, None)
    dataset_group = groups[dataset_name]
    fullname = full_names[dataset_name]

    if verbose:
        print(f"Loading UCI Binary {dataset_name} ({fullname})")

    file = join(data_home, "uci_datasets", dataset_group + ".pkl")

    @contextmanager
    def download_tmp_file(url_group: str, filename: str):
        """
        Download a data file for a group of datasets temporarely.
        When used as a context, the file is removed once the context exits.

        :param url_group: identifier of the dataset group in the URL
        :param filename: name of the file to be downloaded
        """
        data_dir = join(data_home, "uci_datasets", "tmp")
        os.makedirs(data_dir, exist_ok=True)
        data_path = join(data_dir, filename)
        url = f"http://archive.ics.uci.edu/ml/machine-learning-databases/{url_group}/{filename}"
        download_file_if_not_exists(url, data_path)
        try:
            yield data_path
        finally:
            os.remove(data_path)

    def download(id: int | None, group: str) -> dict:
        """
        Download the data to be pickled for a dataset group. Use the `fetch_ucirepo` api when possible.

        :param id: numeric identifier for the group; can be None
        :param group: group name
        :return: a dictionary with X and y as keys and, optionally, extra data.
        """

        # use the fetch_ucirepo api, when possible, to download data
        # fall back to direct download when needed
        if group == "german":
            with download_tmp_file("statlog/german", "german.data-numeric") as tmp:
                df = pd.read_csv(tmp, header=None, delim_whitespace=True)
            X, y = df.iloc[:, 0:24].astype(float).values, df[24].astype(int).values
        elif group == "ctg":
            with download_tmp_file("00193", "CTG.xls") as tmp:
                df = pd.read_excel(tmp, sheet_name="Data", skipfooter=3)
            df = df[list(range(1, 24))]  # select columns numbered (number 23 is the target label)
            # replaces the header with the first row
            new_header = df.iloc[0]  # grab the first row for the header
            df = df[1:]  # take the data less the header row
            df.columns = new_header  # set the header row as the df header
            X = df.iloc[:, 0:21].astype(float).values  # column 21 is skipped, it is a class column
            y = df["NSP"].astype(int).values
        elif group == "semeion":
            with download_tmp_file("semeion", "semeion.data") as tmp:
                df = pd.read_csv(tmp, header=None, delim_whitespace=True)
            X = df.iloc[:, 0:256].astype(float).values
            y = df[263].values  # 263 stands for digit 8 (labels are one-hot vectors from col 256-266)
        else:
            df = fetch_ucirepo(id=id)
            X, y = df.data.features.to_numpy(), df.data.targets.to_numpy().squeeze()

        # transform data when needed before returning (returned data will be pickled)
        if group == "acute":
            _array_replace(X)
            data = {"X": X, "y": y}
        elif group == "balance":
            # features' order is reversed to match data retrieved via direct download
            X = X[:, np.arange(X.shape[1])[::-1]]
            data = {"X": X, "y": y}
        elif group == "breast-cancer":
            # remove rows with nan values
            Xy = np.hstack([X, y[:, np.newaxis]])
            nan_rows = np.isnan(Xy).sum(axis=-1) > 0
            Xy = Xy[~nan_rows]
            data = {"X": Xy[:, :-1], "y": Xy[:, -1]}
        elif group == "mammographic":
            # remove rows with nan values
            Xy = np.hstack([X, y[:, np.newaxis]])
            nan_rows = np.isnan(Xy).sum(axis=-1) > 0
            Xy = Xy[~nan_rows]
            data = {"X": Xy[:, :-1], "y": Xy[:, -1]}
        elif group == "tictactoe":
            _array_replace(X, repl={"o": 0, "b": 1, "x": 2})
            data = {"X": X, "y": y}
        elif group == "wine-quality":
            # add color data to split the final datasets
            color = df.data.original["color"].to_numpy()
            data = {"X": X, "y": y, "color": color}
        else:
            data = {"X": X, "y": y}

        return data

    def binarize_data(name, data: dict) -> LabelledCollection:
        """
        Filter and transform data to extract a binary dataset.

        :param name: name of the dataset
        :param data: dictionary containing X and y fields, plus additional data when needed
        :return: a :class:`quapy.data.base.LabelledCollection` with the extracted dataset
        """
        if name == "acute.a":
            X, y = data["X"], data["y"][:, 0]
            # X, y = Xy[:, :-2], Xy[:, -2]
        elif name == "acute.b":
            X, y = data["X"], data["y"][:, 1]
            # X, y = Xy[:, :-2], Xy[:, -1]
        elif name == "wine-q-red":
            X, y, color = data["X"], data["y"], data["color"]
            # X, y, color = Xy[:, :-2], Xy[:, -2], Xy[:, -1]
            red_idx = color == "red"
            X, y = X[red_idx, :], y[red_idx]
            y = (y > 5).astype(int)
        elif name == "wine-q-white":
            X, y, color = data["X"], data["y"], data["color"]
            # X, y, color = Xy[:, :-2], Xy[:, -2], Xy[:, -1]
            white_idx = color == "white"
            X, y = X[white_idx, :], y[white_idx]
            y = (y > 5).astype(int)
        else:
            X, y = data["X"], data["y"]
            # X, y = Xy[:, :-1], Xy[:, -1]

        y = binarize(y, pos_class=pos_class[name])

        return LabelledCollection(X, y)

    data = pickled_resource(file, download, identifier, dataset_group)
    data = binarize_data(dataset_name, data)

    if standardize:
        stds = StandardScaler()
        data.instances = stds.fit_transform(data.instances)
        
    if verbose:
        data.stats()

    return data


def fetch_UCIMulticlassDataset(
        dataset_name,
        data_home=None,
        min_test_split=0.3,
        max_train_instances=25000,
        min_class_support=100,
        standardize=True,
        verbose=False) -> Dataset:
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
    :param min_test_split: minimum proportion of instances to be included in the test set. This value is interpreted
        as a minimum proportion, meaning that the real proportion could be higher in case the training proportion
        (1-`min_test_split`% of the instances) surpasses `max_train_instances`. In such case, only `max_train_instances`
        are taken for training, and the rest (irrespective of `min_test_split`) is taken for test.
    :param max_train_instances: maximum number of instances to keep for training (defaults to 25000);
        set to -1 or None to avoid this check
    :param min_class_support: minimum number of istances per class. Classes with fewer instances
        are discarded (deafult is 100)
    :param standardize: indicates whether the covariates should be standardized or not (default is True). If requested,
        standardization applies after the LabelledCollection is split, that is, the mean an std are computed only on the
        training portion of the data.
    :param verbose: set to True (default is False) to get information (stats) about the dataset
    :return: a :class:`quapy.data.base.Dataset` instance
    """

    data = fetch_UCIMulticlassLabelledCollection(dataset_name, data_home, min_class_support, verbose=verbose)
    n = len(data)
    train_prop = (1.-min_test_split)
    if (max_train_instances is not None) and (max_train_instances > 0):
        n_train = int(n*train_prop)
        if n_train > max_train_instances:
            train_prop = (max_train_instances / n)

    data = Dataset(*data.split_stratified(train_prop, random_state=0))
    
    if standardize:
        data = standardizer(data)
    
    return data


def fetch_UCIMulticlassLabelledCollection(dataset_name, data_home=None, min_class_support=100, standardize=True, verbose=False) -> LabelledCollection:
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
    :param min_class_support: minimum number of istances per class. Classes with fewer instances
        are discarded (deafult is 100)
    :param standardize: indicates whether the covariates should be standardized or not (default is True). 
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
        'dry-bean': 602,
        'wine-quality': 186,
        'academic-success': 697,
        'digits': 80,
        'letter': 59,
        'abalone': 1,
        'obesity': 544,
        'nursery': 76,
        'yeast': 110,
        'hand_digits': 81,
        'satellite': 146,
        'shuttle': 148,
        'cmc': 30,
        'isolet': 54,
        'waveform-v1': 107,
        'molecular': 69,
        'poker_hand': 158,
        'connect-4': 26,
        'mhr': 863,
        'chess': 23,
        'page_block': 78,
        'phishing': 379,
        'image_seg': 147,
        'hcv': 503,
    }
    
    full_names = {
        'dry-bean': 'Dry Bean Dataset',
        'wine-quality': 'Wine Quality',
        'academic-success': 'Predict students\' dropout and academic success',
        'digits': 'Optical Recognition of Handwritten Digits',
        'letter': 'Letter Recognition',
        'abalone': 'Abalone',
        'obesity': 'Estimation of Obesity Levels Based On Eating Habits and Physical Condition',
        'nursery': 'Nursery',
        'yeast': 'Yeast',
        'hand_digits': 'Pen-Based Recognition of Handwritten Digits',
        'satellite': 'Statlog Landsat Satellite',
        'shuttle': 'Statlog Shuttle',
        'cmc': 'Contraceptive Method Choice',
        'isolet': 'ISOLET',
        'waveform-v1': 'Waveform Database Generator (Version 1)',
        'molecular': 'Molecular Biology (Splice-junction Gene Sequences)',
        'poker_hand': 'Poker Hand',
        'connect-4': 'Connect-4',
        'mhr': 'Maternal Health Risk',
        'chess': 'Chess (King-Rook vs. King)',
        'page_block': 'Page Blocks Classification',
        'phishing': 'Website Phishing',
        'image_seg': 'Statlog (Image Segmentation)',
        'hcv': 'Hepatitis C Virus (HCV) for Egyptian patients',
    }
    
    identifier = identifiers[dataset_name]
    fullname = full_names[dataset_name]

    if verbose:
        print(f'Loading UCI Muticlass {dataset_name} ({fullname})')

    file = join(data_home, 'uci_multiclass', dataset_name+'.pkl')
    
    def download(id, name):
        df = fetch_ucirepo(id=id)

        df.data.features = pd.get_dummies(df.data.features, drop_first=True)
        X, y = df.data.features.to_numpy(dtype=np.float64), df.data.targets.to_numpy().squeeze()

        assert y.ndim == 1, 'more than one y'

        classes = np.sort(np.unique(y))
        y = np.searchsorted(classes, y)
        return LabelledCollection(X, y)

    def filter_classes(data: LabelledCollection, min_ipc):
        if min_ipc is None:
            min_ipc = 0
        classes = data.classes_
        # restrict classes to only those with at least min_ipc instances
        classes = classes[data.counts() >= min_ipc]
        # filter X and y keeping only datapoints belonging to valid classes
        filter_idx = np.in1d(data.y, classes)
        X, y = data.X[filter_idx], data.y[filter_idx]
        # map classes to range(len(classes))
        y = np.searchsorted(classes, y)
        return LabelledCollection(X, y)

    data = pickled_resource(file, download, identifier, dataset_name)
    data = filter_classes(data, min_class_support)
    if data.n_classes <= 2:
        raise ValueError(
            f'After filtering out classes with less than {min_class_support=} instances, the dataset {dataset_name} '
            f'is no longer multiclass. Try a reducing this value.'
        )

    if standardize:
        stds = StandardScaler()
        data.instances = stds.fit_transform(data.instances)

    if verbose:
        data.stats()
        
    return data


def _df_replace(df, col, repl={'yes': 1, 'no':0}, astype=float):
    df[col] = df[col].apply(lambda x:repl[x]).astype(astype, copy=False)


def _array_replace(arr, repl={"yes": 1, "no": 0}):
    for k, v in repl.items():
        arr[arr == k] = v


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

    See `4.lequa2022_experiments.py` provided in the example folder, that can serve as a guide on how to use these
    datasets.

    :param task: a string representing the task name; valid ones are T1A, T1B, T2A, and T2B
    :param data_home: specify the quapy home directory where collections will be dumped (leave empty to use the default
        ~/quay_data/ directory)
    :return: a tuple `(train, val_gen, test_gen)` where `train` is an instance of
        :class:`quapy.data.base.LabelledCollection`, `val_gen` and `test_gen` are instances of
        :class:`quapy.data._lequa2022.SamplesFromDir`, a subclass of :class:`quapy.protocol.AbstractProtocol`,
        that return a series of samples stored in a directory which are labelled by prevalence.
    """

    from quapy.data._lequa import load_raw_documents, load_vector_documents_2022, SamplesFromDir

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
        load_fn = load_vector_documents_2022
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


def fetch_lequa2024(task, data_home=None, merge_T3=False):

    from quapy.data._lequa import load_vector_documents_2024, SamplesFromDir, LabelledCollectionsFromDir

    assert task in LEQUA2024_TASKS, \
        f'Unknown task {task}. Valid ones are {LEQUA2024_TASKS}'

    if data_home is None:
        data_home = get_quapy_home()

    lequa_dir = data_home

    LEQUA2024_ZENODO = 'https://zenodo.org/records/11661820'  # v3, last one with labels

    URL_TRAINDEV=f'{LEQUA2024_ZENODO}/files/{task}.train_dev.zip'
    URL_TEST=f'{LEQUA2024_ZENODO}/files/{task}.test.zip'
    URL_TEST_PREV=f'{LEQUA2024_ZENODO}/files/{task}.test_prevalences.zip'

    lequa_dir = join(data_home, 'lequa2024')
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

    load_fn = load_vector_documents_2024

    val_samples_path = join(lequa_dir, task, 'public', 'dev_samples')
    val_true_prev_path = join(lequa_dir, task, 'public', 'dev_prevalences.txt')
    val_gen = SamplesFromDir(val_samples_path, val_true_prev_path, load_fn=load_fn)

    test_samples_path = join(lequa_dir, task, 'public', 'test_samples')
    test_true_prev_path = join(lequa_dir, task, 'public', 'test_prevalences.txt')
    test_gen = SamplesFromDir(test_samples_path, test_true_prev_path, load_fn=load_fn)

    if task != 'T3':
        tr_path = join(lequa_dir, task, 'public', 'training_data.txt')
        train = LabelledCollection.load(tr_path, loader_func=load_fn)
        return train, val_gen, test_gen
    else:
        training_samples_path = join(lequa_dir, task, 'public', 'training_samples')
        training_true_prev_path = join(lequa_dir, task, 'public', 'training_prevalences.txt')
        train_gen = LabelledCollectionsFromDir(training_samples_path, training_true_prev_path, load_fn=load_fn)
        if merge_T3:
            train = LabelledCollection.join(*list(train_gen()))
            return train, val_gen, test_gen
        else:
            return train_gen, val_gen, test_gen



def fetch_IFCB(single_sample_train=True, for_model_selection=False, data_home=None):
    """
    Loads the IFCB dataset for quantification from `Zenodo <https://zenodo.org/records/10036244>`_ (for more
    information on this dataset, please follow the zenodo link).
    This dataset is based on the data available publicly at
    `WHOI-Plankton repo <https://github.com/hsosik/WHOI-Plankton>`_.
    The dataset already comes with processed features.
    The scripts used for the processing are available at `P. González's repo <https://github.com/pglez82/IFCB_Zenodo>`_.

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
        train, test = generate_modelselection_split(samples, test_prop=0.3)
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
