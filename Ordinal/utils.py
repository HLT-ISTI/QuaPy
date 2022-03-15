import quapy as qp
from quapy.data import LabelledCollection
from glob import glob
import os
from os.path import join
import pickle


def load_samples(path_dir, classes):
    nsamples = len(glob(join(path_dir, f'*.txt')))
    for id in range(nsamples):
        yield LabelledCollection.load(join(path_dir, f'{id}.txt'), loader_func=qp.data.reader.from_text, classes=classes)


def load_samples_as_csv(path_dir, debug=False):
    import pandas as pd
    import csv
    import datasets
    from datasets import Dataset

    nsamples = len(glob(join(path_dir, f'*.txt')))
    for id in range(nsamples):
        df = pd.read_csv(join(path_dir, f'{id}.txt'), sep='\t', names=['labels', 'review'], quoting=csv.QUOTE_NONE)
        labels = df.pop('labels').to_frame()
        X = df

        features = datasets.Features({'review': datasets.Value('string')})
        if debug:
            sample = Dataset.from_pandas(df=X, features=features).select(range(50))
            labels = labels[:50]
        else:
            sample = Dataset.from_pandas(df=X, features=features)

        yield sample, labels


def load_samples_pkl(path_dir, filter=None):
    nsamples = len(glob(join(path_dir, f'*.pkl')))
    for id in range(nsamples):
        if (filter is None) or id in filter:
            yield pickle.load(open(join(path_dir, f'{id}.pkl'), 'rb'))

