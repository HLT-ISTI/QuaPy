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
        for line in fin:
            category, code = line.split()
            cat2code[category] = int(code)
    code2cat = [cat for cat, code in sorted(cat2code.items(), key=lambda x:x[1])]
    return cat2code, code2cat


def load_binary_vectors(path, nF=None):
    return sklearn.datasets.load_svmlight_file(path, n_features=nF)


def __gen_load_samples_with_groudtruth(path_dir:str, return_id:bool, ground_truth_path:str, load_fn, **load_kwargs):
    true_prevs = ResultSubmission.load(ground_truth_path)
    for id, prevalence in true_prevs.iterrows():
        sample, _ = load_fn(os.path.join(path_dir, f'{id}.txt'), **load_kwargs)
        if return_id:
            yield id, sample, prevalence
        else:
            yield sample, prevalence


def __gen_load_samples_without_groudtruth(path_dir:str, return_id:bool, load_fn, **load_kwargs):
    nsamples = len(glob(os.path.join(path_dir, '*.txt')))
    for id in range(nsamples):
        sample, _ = load_fn(os.path.join(path_dir, f'{id}.txt'), **load_kwargs)
        if return_id:
            yield id, sample
        else:
            yield sample


def gen_load_samples_T1(path_dir:str, nF:int, ground_truth_path:str = None, return_id=True):
    if ground_truth_path is None:
        # the generator function returns tuples (filename:str, sample:csr_matrix)
        gen_fn = __gen_load_samples_without_groudtruth(path_dir, return_id, load_binary_vectors, nF=nF)
    else:
        # the generator function returns tuples (filename:str, sample:csr_matrix, prevalence:ndarray)
        gen_fn = __gen_load_samples_with_groudtruth(path_dir, return_id, ground_truth_path, load_binary_vectors, nF=nF)
    for r in gen_fn:
        yield r


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
        self.df = pd.DataFrame(columns=list(categories))
        self.df.index.rename('id', inplace=True)

    def add(self, sample_id:int, prevalence_values:np.ndarray):
        if not isinstance(sample_id, int):
            raise TypeError(f'error: expected int for sample_sample, found {type(sample_id)}')
        if not isinstance(prevalence_values, np.ndarray):
            raise TypeError(f'error: expected np.ndarray for prevalence_values, found {type(prevalence_values)}')
        if sample_id in self.df.index.values:
            raise ValueError(f'error: prevalence values for "{sample_id}" already added')
        if prevalence_values.ndim!=1 and prevalence_values.size != len(self.categories):
            raise ValueError(f'error: wrong shape found for prevalence vector {prevalence_values}')
        if (prevalence_values<0).any() or (prevalence_values>1).any():
            raise ValueError(f'error: prevalence values out of range [0,1] for "{sample_id}"')
        if np.abs(prevalence_values.sum()-1) > constants.ERROR_TOL:
            raise ValueError(f'error: prevalence values do not sum up to one for "{sample_id}"'
                             f'(error tolerance {constants.ERROR_TOL})')

        # new_entry = dict([('id', sample_id)] + [(col_i, prev_i) for col_i, prev_i in enumerate(prevalence_values)])
        new_entry = pd.DataFrame(prevalence_values.reshape(1,2), index=[sample_id], columns=self.df.columns)
        self.df = self.df.append(new_entry, ignore_index=False)

    def __len__(self):
        return len(self.df)

    @classmethod
    def load(cls, path: str) -> 'ResultSubmission':
        df = ResultSubmission.check_file_format(path)
        r = ResultSubmission(categories=df.columns.values.tolist())
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
            # filename = row.filename
            prevalence = row.values.flatten()
            yield index, prevalence

    @classmethod
    def check_file_format(cls, path) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
        df = pd.read_csv(path, index_col=0)
        return ResultSubmission.check_dataframe_format(df, path=path)

    @classmethod
    def check_dataframe_format(cls, df, path=None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
        hint_path = ''  # if given, show the data path in the error message
        if path is not None:
            hint_path = f' in {path}'

        if df.index.name != 'id' or len(df.columns) < 2:
            raise ValueError(f'wrong header{hint_path}, '
                             f'the format of the header should be "id,<cat_1>,...,<cat_n>"')
        if [int(ci) for ci in df.columns.values] != list(range(len(df.columns))):
            raise ValueError(f'wrong header{hint_path}, category ids should be 0,1,2,...,n')

        if df.empty:
            raise ValueError(f'error{hint_path}: results file is empty')
        elif len(df) != constants.DEV_SAMPLES and len(df) != constants.TEST_SAMPLES:
            raise ValueError(f'wrong number of prevalence values found{hint_path}; '
                             f'expected {constants.DEV_SAMPLES} for development sets and '
                             f'{constants.TEST_SAMPLES} for test sets; found {len(df)}')

        ids = set(df.index.values)
        expected_ids = set(range(len(df)))
        if ids != expected_ids:
            missing = expected_ids - ids
            if missing:
                raise ValueError(f'there are {len(missing)} missing ids{hint_path}: {sorted(missing)}')
            unexpected = ids - expected_ids
            if unexpected:
                raise ValueError(f'there are {len(missing)} unexpected ids{hint_path}: {sorted(unexpected)}')

        for category_name in df.columns:
            if (df[category_name] < 0).any() or (df[category_name] > 1).any():
                raise ValueError(f'error{hint_path} column "{category_name}" contains values out of range [0,1]')

        prevs = df.values
        round_errors = np.abs(prevs.sum(axis=-1) - 1.) > constants.ERROR_TOL
        if round_errors.any():
            raise ValueError(f'warning: prevalence values in rows with id {np.where(round_errors)[0].tolist()} '
                              f'do not sum up to 1 (error tolerance {constants.ERROR_TOL}), '
                              f'probably due to some rounding errors.')

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









