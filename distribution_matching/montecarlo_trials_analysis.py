import os
import pickle

from sklearn.linear_model import LogisticRegression
import quapy as qp
from distribution_matching.method.kdey import KDEyHD
from quapy.protocol import UPP
from time import time
import numpy as np
import pandas as pd

qp.environ['SAMPLE_SIZE'] = 500
qp.environ['N_JOBS'] = -1
n_bags_test = 100

os.makedirs('./montecarlo_trials', exist_ok=True)

#take the best bandwidth for each dataset in terms of MAE (as observed in the experiment "sensitivity")
df = pd.read_csv('./results/ucimulti/sensibility/KDEy-ML.csv', sep='\t')
min_mae_bandwidth = df.loc[df.groupby('Dataset')['MAE'].idxmin()]
print(min_mae_bandwidth[['Dataset', 'Bandwidth', 'MAE']])
bandwidth_dict = min_mae_bandwidth.set_index('Dataset')['Bandwidth'].to_dict()

for dataset in qp.datasets.UCI_MULTICLASS_DATASETS:
    print(f'starting {dataset}')
    outfile = f'./montecarlo_trials/{dataset}.pkl'
    if os.path.exists(outfile):
        print(f'{outfile} exists; skipping')
        continue

    train, test = qp.datasets.fetch_UCIMulticlassDataset(dataset).train_test
    train, val = train.split_stratified(0.5, random_state=0)
    lr = LogisticRegression()
    lr.fit(*train.Xy)

    print('\tfit completed')
    test_ae = {}
    times = {}
    for trials in [50, 100, 500, 1000, 5000, 10000, 50000]:

        kdey_hd = KDEyHD(classifier=lr, montecarlo_trials=trials, bandwidth=bandwidth_dict[dataset])

        tinit = time()
        kdey_hd.fit(val, fit_classifier=False, val_split=val)
        trtime = time() - tinit

        tinit = time()
        ae = qp.evaluation.evaluate(kdey_hd, protocol=UPP(test, repeats=n_bags_test, random_state=0), error_metric='ae', verbose=True)
        tetime = (time()-tinit)/n_bags_test

        mae = np.mean(ae)
        mae_std = np.std(ae)
        test_ae[trials] = (mae, mae_std)
        times[trials] = (trtime, tetime)

        print(f'\t{trials=}\tMAE={mae:.4f}+-{mae_std:.4f}\ttraining took {trtime:.3f}\t test took {tetime:.3f}s per sample')

    pickle.dump((test_ae, times), open(f'./montecarlo_trials/{dataset}.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)



