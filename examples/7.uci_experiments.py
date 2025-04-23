from copy import deepcopy

import quapy as qp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from quapy.classification.methods import LowRankLogisticRegression
from quapy.method.meta import QuaNet
from quapy.protocol import APP
from quapy.method.aggregative import CC, ACC, PCC, PACC, MAX, MS, MS2, EMQ, HDy, newSVMAE
from quapy.method.meta import EHDy
import numpy as np
import os
import pickle
import itertools
import argparse
import torch
import shutil


N_JOBS = -1
CUDA_N_JOBS = 2
ENSEMBLE_N_JOBS = -1

qp.environ['SAMPLE_SIZE'] = 100


def newLR():
    return LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)


def calibratedLR():
    return CalibratedClassifierCV(newLR())


__C_range = np.logspace(-3, 3, 7)
lr_params = {
    'classifier__C': __C_range,
    'classifier__class_weight': [None, 'balanced']
}
svmperf_params = {
    'classifier__C': __C_range
}


def quantification_models():
    yield 'cc', CC(newLR()), lr_params
    yield 'acc', ACC(newLR()), lr_params
    yield 'pcc', PCC(newLR()), lr_params
    yield 'pacc', PACC(newLR()), lr_params
    yield 'MAX', MAX(newLR()), lr_params
    yield 'MS', MS(newLR()), lr_params
    yield 'MS2', MS2(newLR()), lr_params
    yield 'sldc', EMQ(newLR(), calib='platt'), lr_params
    yield 'svmmae', newSVMAE(), svmperf_params
    yield 'hdy', HDy(newLR()), lr_params


def quantification_cuda_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running QuaNet in {device}')
    learner = LowRankLogisticRegression()
    yield 'quanet', QuaNet(learner, checkpointdir=args.checkpointdir, device=device), lr_params


def evaluate_experiment(true_prevalences, estim_prevalences):
    print('\nEvaluation Metrics:\n' + '=' * 22)
    for eval_measure in [qp.error.mae, qp.error.mrae]:
        err = eval_measure(true_prevalences, estim_prevalences)
        print(f'\t{eval_measure.__name__}={err:.4f}')
    print()


def result_path(path, dataset_name, model_name, run, optim_loss):
    return os.path.join(path, f'{dataset_name}-{model_name}-run{run}-{optim_loss}.pkl')


def is_already_computed(dataset_name, model_name, run, optim_loss):
    return os.path.exists(result_path(args.results, dataset_name, model_name, run, optim_loss))


def save_results(dataset_name, model_name, run, optim_loss, *results):
    rpath = result_path(args.results, dataset_name, model_name, run, optim_loss)
    qp.util.create_parent_dir(rpath)
    with open(rpath, 'wb') as foo:
        pickle.dump(tuple(results), foo, pickle.HIGHEST_PROTOCOL)


def run(experiment):
    optim_loss, dataset_name, (model_name, model, hyperparams) = experiment
    if dataset_name in ['acute.a', 'acute.b', 'iris.1']: return

    collection = qp.datasets.fetch_UCIBinaryLabelledCollection(dataset_name)
    for run, data in enumerate(qp.data.Dataset.kFCV(collection, nfolds=5, nrepeats=1)):
        if is_already_computed(dataset_name, model_name, run=run, optim_loss=optim_loss):
            print(f'result for dataset={dataset_name} model={model_name} loss={optim_loss} run={run+1}/5 already computed.')
            continue

        print(f'running dataset={dataset_name} model={model_name} loss={optim_loss} run={run+1}/5')
        # model selection (hyperparameter optimization for a quantification-oriented loss)
        train, test = data.train_test
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
            best_params = model_selection.best_params_
        else:
            model.fit(data.training)
            best_params = {}

        # model evaluation
        true_prevalences, estim_prevalences = qp.evaluation.prediction(
            model,
            protocol=APP(test, n_prevalences=21, repeats=100)
        )
        test_true_prevalence = data.test.prevalence()

        evaluate_experiment(true_prevalences, estim_prevalences)
        save_results(dataset_name, model_name, run, optim_loss,
                     true_prevalences, estim_prevalences,
                     data.training.prevalence(), test_true_prevalence,
                     best_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments for Tweeter Sentiment Quantification')
    parser.add_argument('results', metavar='RESULT_PATH', type=str,
                        help='path to the directory where to store the results')
    parser.add_argument('--svmperfpath', metavar='SVMPERF_PATH', type=str, default='../svm_perf_quantification',
                        help='path to the directory with svmperf')
    parser.add_argument('--checkpointdir', metavar='PATH', type=str, default='./checkpoint',
                        help='path to the directory where to dump QuaNet checkpoints')
    args = parser.parse_args()

    print(f'Result folder: {args.results}')
    np.random.seed(0)

    qp.environ['SVMPERF_HOME'] = args.svmperfpath

    optim_losses = ['mae']
    datasets = qp.datasets.UCI_BINARY_DATASETS

    models = quantification_models()
    qp.util.parallel(run, itertools.product(optim_losses, datasets, models), n_jobs=N_JOBS)

    models = quantification_cuda_models()
    qp.util.parallel(run, itertools.product(optim_losses, datasets, models), n_jobs=CUDA_N_JOBS)

    shutil.rmtree(args.checkpointdir, ignore_errors=True)
