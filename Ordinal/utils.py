import quapy as qp
from quapy.data import LabelledCollection
from glob import glob
import os
from os.path import join


def load_samples(path_dir, classes):
    nsamples = len(glob(join(path_dir, f'*.txt')))
    for id in range(nsamples):
        yield LabelledCollection.load(join(path_dir, f'{id}.txt'), loader_func=qp.data.reader.from_text, classes=classes)


