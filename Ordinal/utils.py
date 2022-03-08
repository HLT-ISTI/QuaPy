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


def load_samples_pkl(path_dir, filter=None):
    nsamples = len(glob(join(path_dir, f'*.pkl')))
    for id in range(nsamples):
        if filter is not None:
            if id not in filter:
                continue
        yield pickle.load(open(join(path_dir, f'{id}.pkl'), 'rb'))

