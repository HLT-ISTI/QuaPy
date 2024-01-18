from copy import deepcopy

import quapy as qp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from quapy.classification.methods import LowRankLogisticRegression
from quapy.method.meta import QuaNet
from quapy.protocol import APP
from quapy.method.aggregative import CC, ACC, PCC, PACC, MAX, MS, MS2, EMQ, HDy, newSVMAE, T50, X
from quapy.method.meta import EHDy
import numpy as np
import os
import pickle
import itertools
import argparse
from glob import glob
import pandas as pd
from time import time

N_JOBS = -1

qp.environ['SAMPLE_SIZE'] = 100


def newLR():
    return LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)


def calibratedLR():
    return CalibratedClassifierCV(LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1))


__C_range = np.logspace(-3, 3, 7)
lr_params = {'classifier__C': __C_range, 'classifier__class_weight': [None, 'balanced']}
svmperf_params = {'classifier__C': __C_range}


def quantification_models():
    yield 'acc', ACC(newLR()), lr_params
    yield 'T50', T50(newLR()), lr_params
    yield 'X', X(newLR()), lr_params
    yield 'MAX', MAX(newLR()), lr_params
    yield 'MS', MS(newLR()), lr_params
    yield 'MS+', MS(newLR()), lr_params
    # yield 'MS2', MS2(newLR()), lr_params



def result_path(path, dataset_name, model_name, optim_loss):
    return os.path.join(path, f'{dataset_name}-{model_name}-{optim_loss}.pkl')


def is_already_computed(dataset_name, model_name, optim_loss):
    return os.path.exists(result_path(args.results, dataset_name, model_name, optim_loss))


def save_results(dataset_name, model_name, optim_loss, *results):
    rpath = result_path(args.results, dataset_name, model_name, optim_loss)
    qp.util.create_parent_dir(rpath)
    with open(rpath, 'wb') as foo:
        pickle.dump(tuple(results), foo, pickle.HIGHEST_PROTOCOL)


def run(experiment):
    optim_loss, dataset_name, (model_name, model, hyperparams) = experiment
    if dataset_name in ['acute.a', 'acute.b', 'iris.1']: return

    if is_already_computed(dataset_name, model_name, optim_loss=optim_loss):
        print(f'result for dataset={dataset_name} model={model_name} loss={optim_loss} already computed.')
        return

    dataset = qp.datasets.fetch_UCIDataset(dataset_name)

    print(f'running dataset={dataset_name} model={model_name} loss={optim_loss}')
    # model selection (hyperparameter optimization for a quantification-oriented loss)
    train, test = dataset.train_test
    train, val = train.split_stratified()
    if hyperparams is not None:
        model_selection = qp.model_selection.GridSearchQ(
            deepcopy(model),
            param_grid=hyperparams,
            protocol=APP(val, n_prevalences=21, repeats=25),
            error=optim_loss,
            refit=True,
            timeout=60*60,
            verbose=True
        )
        model_selection.fit(train)
        model = model_selection.best_model()
    else:
        model.fit(dataset.training)

    # model evaluation
    true_prevalences, estim_prevalences = qp.evaluation.prediction(
        model,
        protocol=APP(test, n_prevalences=21, repeats=100)
    )

    mae = qp.error.mae(true_prevalences, estim_prevalences)
    save_results(dataset_name, model_name, optim_loss, mae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments for Tweeter Sentiment Quantification')
    parser.add_argument('--results', metavar='RESULT_PATH', type=str, default='results_tmp',
                        help='path to the directory where to store the results')
    parser.add_argument('--svmperfpath', metavar='SVMPERF_PATH', type=str, default='../svm_perf_quantification',
                        help='path to the directory with svmperf')
    args = parser.parse_args()

    print(f'Result folder: {args.results}')
    np.random.seed(0)

    qp.environ['SVMPERF_HOME'] = args.svmperfpath

    optim_losses = ['mae']
    datasets = qp.datasets.UCI_DATASETS

    tstart = time()
    models = quantification_models()
    qp.util.parallel(run, itertools.product(optim_losses, datasets, models), n_jobs=N_JOBS)
    tend = time()

    # open all results and show
    df = pd.DataFrame(columns=('method', 'dataset', 'mae'))
    for i, file in enumerate(glob(f'{args.results}/*.pkl')):
        mae = float(pickle.load(open(file, 'rb'))[0])
        *dataset, method, _ = file.split('/')[-1].split('-')
        dataset = '-'.join(dataset)
        df.loc[i] = [method, dataset, mae]

    print(df.pivot_table(index='dataset', columns='method', values='mae', margins=True))

    print(f'took {(tend-tstart)}s')
