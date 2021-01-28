import numpy as np
import quapy as qp
import settings
import os
import pickle
from glob import glob
import itertools
import pathlib

qp.environ['SAMPLE_SIZE'] = settings.SAMPLE_SIZE

resultdir = './results'
methods = ['*']


def evaluate_results(methods, datasets, error_name):
    results_str = []
    all = []
    error = qp.error.from_name(error_name)
    for method, dataset in itertools.product(methods, datasets):
        for experiment in glob(f'{resultdir}/{dataset}-{method}-{error_name}.pkl'):
            true_prevalences, estim_prevalences, tr_prev, te_prev, te_prev_estim, best_params = \
                pickle.load(open(experiment, 'rb'))
            result = error(true_prevalences, estim_prevalences)
            string = f'{pathlib.Path(experiment).name}: {result:.3f}'
            results_str.append(string)
            all.append(result)
    results_str = sorted(results_str)
    for r in results_str:
        print(r)
    print()
    print(f'Ave: {np.mean(all):.3f}')


evaluate_results(methods=['epacc*mae1k'], datasets=['*'], error_name='mae')
