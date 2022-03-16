import numpy as np
import quapy as qp
from evaluation import nmd
from Ordinal.utils import load_samples_folder, load_single_sample_pkl
from quapy.data import LabelledCollection
import pickle
import os
from os.path import join
from tqdm import tqdm


def partition_by_drift(split, training_prevalence):
    assert split in ['dev', 'test'], 'invalid split name'
    total=1000 if split=='dev' else 5000
    drifts = []
    folderpath = join(datapath, domain, 'app', f'{split}_samples')
    for sample in tqdm(load_samples_folder(folderpath, load_fn=load_single_sample_pkl), total=total):
        drifts.append(nmd(training_prevalence, sample.prevalence()))
    drifts = np.asarray(drifts)
    order = np.argsort(drifts)
    nD = len(order)
    low_drift, mid_drift, high_drift = order[:nD // 3], order[nD // 3:2 * nD // 3], order[2 * nD // 3:]
    all_drift = np.arange(nD)
    np.save(join(datapath, domain, 'app', f'lowdrift.{split}.id.npy'), low_drift)
    np.save(join(datapath, domain, 'app', f'middrift.{split}.id.npy'), mid_drift)
    np.save(join(datapath, domain, 'app', f'highdrift.{split}.id.npy'), high_drift)
    np.save(join(datapath, domain, 'app', f'alldrift.{split}.id.npy'), all_drift)
    lows = drifts[low_drift]
    mids = drifts[mid_drift]
    highs = drifts[high_drift]
    all = drifts[all_drift]
    print(f'low drift: interval [{lows.min():.4f}, {lows.max():.4f}] mean: {lows.mean():.4f}')
    print(f'mid drift: interval [{mids.min():.4f}, {mids.max():.4f}] mean: {mids.mean():.4f}')
    print(f'high drift: interval [{highs.min():.4f}, {highs.max():.4f}] mean: {highs.mean():.4f}')
    print(f'all drift: interval [{all.min():.4f}, {all.max():.4f}] mean: {all.mean():.4f}')


domain = 'Books-roberta-base-finetuned-pkl/checkpoint-1188-average'
datapath = './data'

training = pickle.load(open(join(datapath,domain,'training_data.pkl'), 'rb'))

partition_by_drift('dev', training.prevalence())
partition_by_drift('test', training.prevalence())

