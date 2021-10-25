import os.path
from typing import List, Tuple, Union

import pandas as pd

import quapy as qp
import numpy as np
import sklearn
import re
from glob import glob

import constants


# def load_binary_raw_document(path):
#     documents, labels = qp.data.from_text(path, verbose=0, class2int=True)
#     labels = np.asarray(labels)
#     labels[np.logical_or(labels == 1, labels == 2)] = 0
#     labels[np.logical_or(labels == 4, labels == 5)] = 1
#     return documents, labels


# def load_multiclass_raw_document(path):
#     return qp.data.from_text(path, verbose=0, class2int=False)

def load_category_map(path):
    cat2code = {}
    with open(path, 'rt') as fin:
        category, code = fin.readline().split()
        cat2code[category] = int(code)
    return cat2code


def load_binary_vectors(path, nF=None):
    return sklearn.datasets.load_svmlight_file(path, n_features=nF)


def __gen_load_samples_with_groudtruth(path_dir:str, ground_truth_path:str, load_fn, **load_kwargs):
    true_prevs = ResultSubmission.load(ground_truth_path)
    for filename, prevalence in true_prevs.iterrows():
        sample, _ = load_fn(os.path.join(path_dir, filename), **load_kwargs)
        yield filename, sample, prevalence


def __gen_load_samples_without_groudtruth(path_dir:str, load_fn, **load_kwargs):
    for filepath in glob(os.path.join(path_dir, '*_sample_*.txt')):
        sample, _ = load_fn(filepath, **load_kwargs)
        yield os.path.basename(filepath), sample


def gen_load_samples_T1A(path_dir:str, nF:int, ground_truth_path:str = None):
    if ground_truth_path is None:
        for filename, sample in __gen_load_samples_without_groudtruth(path_dir, load_binary_vectors, nF=nF):
            yield filename, sample
    else:
        for filename, sample, prevalence in __gen_load_samples_with_groudtruth(path_dir, ground_truth_path, load_binary_vectors, nF=nF):
            yield filename, sample, prevalence


def gen_load_samples_T1B(path_dir:str, ground_truth_path:str = None):
    # for ... : yield
    pass


def gen_load_samples_T2A(path_dir:str, ground_truth_path:str = None):
    # for ... : yield
    pass


def gen_load_samples_T2B(path_dir:str, ground_truth_path:str = None):
    # for ... : yield
    pass


class ResultSubmission:

    def __init__(self, categories: List[str]):
        if not isinstance(categories, list) or len(categories) < 2:
            raise TypeError('wrong format for categories; a list with at least two category names (str) was expected')
        self.categories = categories
        self.df = pd.DataFrame(columns=['filename'] + list(categories))
        self.inferred_type = None

    def add(self, sample_name:str, prevalence_values:np.ndarray):
        if not isinstance(sample_name, str):
            raise TypeError(f'error: expected str for sample_sample, found {type(sample_name)}')
        if not isinstance(prevalence_values, np.ndarray):
            raise TypeError(f'error: expected np.ndarray for prevalence_values, found {type(prevalence_values)}')

        if self.inferred_type is None:
            if sample_name.startswith('test'):
                self.inferred_type = 'test'
            elif sample_name.startswith('dev'):
                self.inferred_type = 'dev'
        else:
            if not sample_name.startswith(self.inferred_type):
                raise ValueError(f'error: sample "{sample_name}" is not a valid entry for type "{self.inferred_type}"')

        if not re.match("(test|dev)_sample_\d+\.txt", sample_name):
            raise ValueError(f'error: wrong format "{sample_name}"; right format is (test|dev)_sample_<number>.txt')
        if sample_name in self.df.filename.values:
            raise ValueError(f'error: prevalence values for "{sample_name}" already added')
        if prevalence_values.ndim!=1 and prevalence_values.size != len(self.categories):
            raise ValueError(f'error: wrong shape found for prevalence vector {prevalence_values}')
        if (prevalence_values<0).any() or (prevalence_values>1).any():
            raise ValueError(f'error: prevalence values out of range [0,1] for "{sample_name}"')
        if np.abs(prevalence_values.sum()-1) > constants.ERROR_TOL:
            raise ValueError(f'error: prevalence values do not sum up to one for "{sample_name}"'
                             f'(error tolerance {constants.ERROR_TOL})')

        new_entry = dict([('filename',sample_name)]+[(col_i,prev_i) for col_i, prev_i in zip(self.categories, prevalence_values)])
        self.df = self.df.append(new_entry, ignore_index=True)

    def __len__(self):
        return len(self.df)

    @classmethod
    def load(cls, path: str) -> 'ResultSubmission':
        df, inferred_type = ResultSubmission.check_file_format(path, return_inferred_type=True)
        r = ResultSubmission(categories=df.columns.values[1:].tolist())
        r.inferred_type = inferred_type
        r.df = df
        return r

    def dump(self, path:str):
        ResultSubmission.check_dataframe_format(self.df)
        self.df.to_csv(path)

    def prevalence(self, sample_name:str):
        sel = self.df.loc[self.df['filename'] == sample_name]
        if sel.empty:
            return None
        else:
            return sel.loc[:,self.df.columns[1]:].values.flatten()

    def iterrows(self):
        for index, row in self.df.iterrows():
            filename = row.filename
            prevalence = row[self.df.columns[1]:].values.flatten()
            yield filename, prevalence

    @classmethod
    def check_file_format(cls, path, return_inferred_type=False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
        df = pd.read_csv(path, index_col=0)
        return ResultSubmission.check_dataframe_format(df, path=path, return_inferred_type=return_inferred_type)

    @classmethod
    def check_dataframe_format(cls, df, path=None, return_inferred_type=False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
        hint_path = ''  # if given, show the data path in the error message
        if path is not None:
            hint_path = f' in {path}'

        if 'filename' not in df.columns or len(df.columns) < 3:
            raise ValueError(f'wrong header{hint_path}, the format of the header should be ",filename,<cat_1>,...,<cat_n>"')

        if df.empty:
            raise ValueError(f'error{hint_path}: results file is empty')
        elif len(df) == constants.DEV_SAMPLES:
            inferred_type = 'dev'
            expected_len = constants.DEV_SAMPLES
        elif len(df) == constants.TEST_SAMPLES:
            inferred_type = 'test'
            expected_len = constants.TEST_SAMPLES
        else:
            raise ValueError(f'wrong number of prevalence values found{hint_path}; '
                             f'expected {constants.DEV_SAMPLES} for development sets and '
                             f'{constants.TEST_SAMPLES} for test sets; found {len(df)}')

        set_names = frozenset(df.filename)
        for i in range(expected_len):
            if f'{inferred_type}_sample_{i}.txt' not in set_names:
                raise ValueError(f'error{hint_path} a file with {len(df)} entries is assumed to be of type '
                                 f'"{inferred_type}" but entry {inferred_type}_sample_{i}.txt is missing '
                                 f'(among perhaps many others)')

        for category_name in df.columns[1:]:
            if (df[category_name] < 0).any() or (df[category_name] > 1).any():
                raise ValueError(f'error{hint_path} column "{category_name}" contains values out of range [0,1]')

        prevs = df.loc[:, df.columns[1]:].values
        round_errors = np.abs(prevs.sum(axis=-1) - 1.) > constants.ERROR_TOL
        if round_errors.any():
            raise ValueError(f'warning: prevalence values in rows with id {np.where(round_errors)[0].tolist()} '
                              f'do not sum up to 1 (error tolerance {constants.ERROR_TOL}), '
                              f'probably due to some rounding errors.')

        if return_inferred_type:
            return df, inferred_type
        else:
            return df

    def sort_categories(self):
        self.df = self.df.reindex([self.df.columns[0]] + sorted(self.df.columns[1:]), axis=1)
        self.categories = sorted(self.categories)

    def filenames(self):
        return self.df.filename.values


def evaluate_submission(true_prevs: ResultSubmission, predicted_prevs: ResultSubmission, sample_size=None, average=True):
    if sample_size is None:
        if qp.environ['SAMPLE_SIZE'] is None:
            raise ValueError('Relative Absolute Error cannot be computed: '
                             'neither sample_size nor qp.environ["SAMPLE_SIZE"] have been specified')
        else:
            sample_size = qp.environ['SAMPLE_SIZE']

    if len(true_prevs) != len(predicted_prevs):
        raise ValueError(f'size mismatch, ground truth file has {len(true_prevs)} entries '
                         f'while the file of predictions contain {len(predicted_prevs)} entries')
    true_prevs.sort_categories()
    predicted_prevs.sort_categories()
    if true_prevs.categories != predicted_prevs.categories:
        raise ValueError(f'these result files are not comparable since the categories are different: '
                         f'true={true_prevs.categories} vs. predictions={predicted_prevs.categories}')
    ae, rae = [], []
    for sample_name, true_prevalence in true_prevs.iterrows():
        pred_prevalence = predicted_prevs.prevalence(sample_name)
        ae.append(qp.error.ae(true_prevalence, pred_prevalence))
        rae.append(qp.error.rae(true_prevalence, pred_prevalence, eps=1./(2*sample_size)))
    ae = np.asarray(ae)
    rae = np.asarray(rae)
    if average:
        return ae.mean(), rae.mean()
    else:
        return ae, rae









