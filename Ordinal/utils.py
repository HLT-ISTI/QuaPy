import numpy as np
from glob import glob
from json import load
import os
from os.path import join
import pickle
import pandas as pd
import csv
import datasets
from datasets import Dataset
import quapy as qp
from quapy.data import LabelledCollection



def load_simple_sample_npytxt(parentdir, filename, classes=None):
    samplepath = join(parentdir, filename+'.txt')
    yX = np.loadtxt(samplepath)
    X = yX[:,1:]
    y = yX[:,0].astype(np.int32)
    return LabelledCollection(instances=X, labels=y, classes_=classes)


def load_simple_sample_raw(parentdir, filename, classes=None):
    samplepath = join(parentdir, filename+'.txt')
    return LabelledCollection.load(samplepath, loader_func=qp.data.reader.from_text, classes=classes)


def load_single_sample_as_csv(parentdir, filename):
    samplepath = join(parentdir, filename+'.txt')
    df = pd.read_csv(samplepath, sep='\t', names=['labels', 'review'], quoting=csv.QUOTE_NONE)
    labels = df.pop('labels').to_frame()

    features = datasets.Features({'review': datasets.Value('string')})
    sample = Dataset.from_pandas(df=df, features=features)

    return sample, labels


def load_single_sample_pkl(parentdir, filename):
    return pickle.load(open(join(parentdir, filename+'.pkl'), 'rb'))


# def load_samples_npytxt(path_dir, filter=None, classes=None):
#     return load_samples_folder(path_dir, filter, load_fn=load_simple_sample_npytxt)


# def load_samples_raw(path_dir, filter=None, classes=None):
#     return load_samples_folder(path_dir, filter, load_fn=load_simple_sample_raw, load_fn_kwargs={'classes': classes})


# def load_samples_as_csv(path_dir, filter=None):
#     return load_samples_folder(path_dir, filter, load_fn=load_single_sample_as_csv)


# def load_samples_pkl(path_dir, filter=None):
#     return load_samples_folder(path_dir, filter, load_fn=load_single_sample_pkl)


def load_samples_folder(path_dir, filter=None, load_fn=None, **load_fn_kwargs):
    nsamples = len(glob(join(path_dir, f'*')))
    for id in range(nsamples):
        if (filter is None) or id in filter:
            yield load_fn(path_dir, f'{id}', **load_fn_kwargs)
