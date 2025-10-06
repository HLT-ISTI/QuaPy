from typing import Tuple, Union
import pandas as pd
import numpy as np
import os

from quapy.protocol import AbstractProtocol
from quapy.data import LabelledCollection


DEV_SAMPLES = 1000
TEST_SAMPLES = 5000

ERROR_TOL = 1E-3


def load_category_map(path):
    """
    Loads the category map, i.e., a mapping of numerical ids of labels with a human readable name.

    :param path: path to the label map file
    :return: a dictionary cat2code (i.e., cat2code[cat_name] gives access to the category id) and a list code2cat (i.e.,
        code2cat[cat_id] gives access to the category name)
    """
    cat2code = {}
    with open(path, 'rt') as fin:
        for line in fin:
            category, code = line.split()
            cat2code[category] = int(code)
    code2cat = [cat for cat, code in sorted(cat2code.items(), key=lambda x: x[1])]
    return cat2code, code2cat


def load_raw_documents(path):
    """
    Loads raw documents. In case the sample is unlabelled,
    the labels returned are None

    :param path: path to the data sample containing the raw documents
    :return: a tuple with the documents (np.ndarray of strings of shape `(n,)`) and
        the labels (a np.ndarray of shape `(n,)` if the sample is labelled,
        or None if the sample is unlabelled), with `n` the number of instances in the sample
        (250 for T1A, 1000 for T1B)
    """
    df = pd.read_csv(path)
    documents = list(df["text"].values)
    labels = None
    if "label" in df.columns:
        labels = df["label"].values.astype(int)
    return documents, labels


def load_vector_documents_2022(path):
    """
    Loads vectorized documents. In case the sample is unlabelled,
    the labels returned are None

    :param path: path to the data sample containing the raw documents
    :return: a tuple with the documents (np.ndarray of shape `(n,300)`) and the labels (a np.ndarray of shape `(n,)` if
        the sample is labelled, or None if the sample is unlabelled), with `n` the number of instances in the sample
        (250 for T1A, 1000 for T1B)
    """
    D = pd.read_csv(path).to_numpy(dtype=float)
    labelled = D.shape[1] == 301
    if labelled:
        X, y = D[:, 1:], D[:, 0].astype(int).flatten()
    else:
        X, y = D, None
    return X, y


def load_vector_documents_2024(path):
    """
    Loads vectorized documents. In case the sample is unlabelled,
    the labels returned are None

    :param path: path to the data sample containing the raw documents
    :return: a tuple with the documents (np.ndarray of shape `(n,256)`) and the labels (a np.ndarray of shape `(n,)` if
        the sample is labelled, or None if the sample is unlabelled), with `n` the number of instances in the sample
        (250 for T1 and T4, 1000 for T2, and 200 for T3)
    """
    D = pd.read_csv(path).to_numpy(dtype=float)
    labelled = D.shape[1] == 257
    if labelled:
        X, y = D[:,1:], D[:,0].astype(int).flatten()
    else:
        X, y = D, None
    return X, y


class SamplesFromDir(AbstractProtocol):

    def __init__(self, path_dir:str, ground_truth_path:str, load_fn):
        self.path_dir = path_dir
        self.load_fn = load_fn
        self.true_prevs = ResultSubmission.load(ground_truth_path)

    def __call__(self):
        for id, prevalence in self.true_prevs.iterrows():
            sample, _ = self.load_fn(os.path.join(self.path_dir, f'{id}.txt'))
            yield sample, prevalence


class LabelledCollectionsFromDir(AbstractProtocol):

    def __init__(self, path_dir:str, ground_truth_path:str, load_fn):
        self.path_dir = path_dir
        self.load_fn = load_fn
        self.true_prevs = pd.read_csv(ground_truth_path, index_col=0)

    def __call__(self):
        for id, prevalence in self.true_prevs.iterrows():
            collection_path = os.path.join(self.path_dir, f'{id}.txt')
            lc = LabelledCollection.load(path=collection_path, loader_func=self.load_fn)
            yield lc


class ResultSubmission:

    def __init__(self):
        self.df = None

    def __init_df(self, categories: int):
        if not isinstance(categories, int) or categories < 2:
            raise TypeError('wrong format for categories: an int (>=2) was expected')
        df = pd.DataFrame(columns=list(range(categories)))
        df.index.set_names('id', inplace=True)
        self.df = df

    @property
    def n_categories(self):
        return len(self.df.columns.values)

    def add(self, sample_id: int, prevalence_values: np.ndarray):
        if not isinstance(sample_id, int):
            raise TypeError(f'error: expected int for sample_sample, found {type(sample_id)}')
        if not isinstance(prevalence_values, np.ndarray):
            raise TypeError(f'error: expected np.ndarray for prevalence_values, found {type(prevalence_values)}')
        if self.df is None:
            self.__init_df(categories=len(prevalence_values))
        if sample_id in self.df.index.values:
            raise ValueError(f'error: prevalence values for "{sample_id}" already added')
        if prevalence_values.ndim != 1 and prevalence_values.size != self.n_categories:
            raise ValueError(f'error: wrong shape found for prevalence vector {prevalence_values}')
        if (prevalence_values < 0).any() or (prevalence_values > 1).any():
            raise ValueError(f'error: prevalence values out of range [0,1] for "{sample_id}"')
        if np.abs(prevalence_values.sum() - 1) > ERROR_TOL:
            raise ValueError(f'error: prevalence values do not sum up to one for "{sample_id}"'
                             f'(error tolerance {ERROR_TOL})')

        self.df.loc[sample_id] = prevalence_values

    def __len__(self):
        return len(self.df)

    @classmethod
    def load(cls, path: str) -> 'ResultSubmission':
        df = ResultSubmission.check_file_format(path)
        r = ResultSubmission()
        r.df = df
        return r

    def dump(self, path: str):
        ResultSubmission.check_dataframe_format(self.df)
        self.df.to_csv(path)

    def prevalence(self, sample_id: int):
        sel = self.df.loc[sample_id]
        if sel.empty:
            return None
        else:
            return sel.values.flatten()

    def iterrows(self):
        for index, row in self.df.iterrows():
            prevalence = row.values.flatten()
            yield index, prevalence

    @classmethod
    def check_file_format(cls, path) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception as e:
            print(f'the file {path} does not seem to be a valid csv file. ')
            print(e)
        return ResultSubmission.check_dataframe_format(df, path=path)

    @classmethod
    def check_dataframe_format(cls, df, path=None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
        hint_path = ''  # if given, show the data path in the error message
        if path is not None:
            hint_path = f' in {path}'

        if df.index.name != 'id' or len(df.columns) < 2:
            raise ValueError(f'wrong header{hint_path}, '
                             f'the format of the header should be "id,0,...,n-1", '
                             f'where n is the number of categories')
        if [int(ci) for ci in df.columns.values] != list(range(len(df.columns))):
            raise ValueError(f'wrong header{hint_path}, category ids should be 0,1,2,...,n-1, '
                             f'where n is the number of categories')
        if df.empty:
            raise ValueError(f'error{hint_path}: results file is empty')
        elif len(df) != DEV_SAMPLES and len(df) != TEST_SAMPLES:
            raise ValueError(f'wrong number of prevalence values found{hint_path}; '
                             f'expected {DEV_SAMPLES} for development sets and '
                             f'{TEST_SAMPLES} for test sets; found {len(df)}')

        ids = set(df.index.values)
        expected_ids = set(range(len(df)))
        if ids != expected_ids:
            missing = expected_ids - ids
            if missing:
                raise ValueError(f'there are {len(missing)} missing ids{hint_path}: {sorted(missing)}')
            unexpected = ids - expected_ids
            if unexpected:
                raise ValueError(f'there are {len(missing)} unexpected ids{hint_path}: {sorted(unexpected)}')

        for category_id in df.columns:
            if (df[category_id] < 0).any() or (df[category_id] > 1).any():
                raise ValueError(f'error{hint_path} column "{category_id}" contains values out of range [0,1]')

        prevs = df.values
        round_errors = np.abs(prevs.sum(axis=-1) - 1.) > ERROR_TOL
        if round_errors.any():
            raise ValueError(f'warning: prevalence values in rows with id {np.where(round_errors)[0].tolist()} '
                             f'do not sum up to 1 (error tolerance {ERROR_TOL}), '
                             f'probably due to some rounding errors.')

        return df


