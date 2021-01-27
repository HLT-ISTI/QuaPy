from sklearn.linear_model import LogisticRegression
import quapy as qp
from classification.methods import PCALR
from method.meta import QuaNet
from method.non_aggregative import MaximumLikelihoodPrevalenceEstimation
from quapy.method.aggregative import CC, ACC, PCC, PACC, EMQ, OneVsAll, SVMQ, SVMKLD, SVMNKLD, SVMAE, SVMRAE, HDy
from quapy.method.meta import EPACC, EEMQ
import quapy.functional as F
import numpy as np
import os
import pickle
import itertools
from joblib import Parallel, delayed
import settings
import argparse
import torch
import shutil


qp.environ['SAMPLE_SIZE'] = settings.SAMPLE_SIZE

def newLR():
    return LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)

__C_range = np.logspace(-4, 5, 10)
lr_params = {'C': __C_range, 'class_weight': [None, 'balanced']}
svmperf_params = {'C': __C_range}

def quantification_models():
    # methods tested in Gao & Sebastiani 2016
    yield 'cc', CC(newLR()), lr_params
    yield 'acc', ACC(newLR()), lr_params
    yield 'pcc', PCC(newLR()), lr_params
    yield 'pacc', PACC(newLR()), lr_params
    yield 'sld', EMQ(newLR()), lr_params
    yield 'svmq', OneVsAll(SVMQ(args.svmperfpath)), svmperf_params
    yield 'svmkld', OneVsAll(SVMKLD(args.svmperfpath)), svmperf_params
    yield 'svmnkld', OneVsAll(SVMNKLD(args.svmperfpath)), svmperf_params

    # methods added
    yield 'svmmae', OneVsAll(SVMAE(args.svmperfpath)), svmperf_params
    yield 'svmmrae', OneVsAll(SVMRAE(args.svmperfpath)), svmperf_params
    yield 'hdy', OneVsAll(HDy(newLR())), lr_params


def quantification_cuda_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running QuaNet in {device}')
    learner = PCALR(**newLR().get_params())
    yield 'quanet', QuaNet(learner, settings.SAMPLE_SIZE, checkpointdir=args.checkpointdir, device=device), lr_params


def quantification_ensembles():
    param_mod_sel = {
        'sample_size': settings.SAMPLE_SIZE,
        'n_prevpoints': 21,
        'n_repetitions': 5,
        'verbose': False
    }
    common={
        'max_sample_size': 500,
        'n_jobs': settings.ENSEMBLE_N_JOBS,
        'param_grid': lr_params,
        'param_mod_sel': param_mod_sel,
        'val_split': 0.4,
        'min_pos': 10
    }
    
    # hyperparameters will be evaluated within each quantifier of the ensemble, and so the typical model selection
    # will be skipped (by setting hyperparameters to None)
    hyper_none = None
    yield 'epaccmaeptr', EPACC(newLR(), optim='mae', policy='ptr', **common), hyper_none
    yield 'epaccmaemae', EPACC(newLR(), optim='mae', policy='mae', **common), hyper_none
    # yield 'esldmaeptr', EEMQ(newLR(), optim='mae', policy='ptr', **common), hyper_none
    # yield 'esldmaemae', EEMQ(newLR(), optim='mae', policy='mae', **common), hyper_none

    yield 'epaccmraeptr', EPACC(newLR(), optim='mrae', policy='ptr', **common), hyper_none
    yield 'epaccmraemrae', EPACC(newLR(), optim='mrae', policy='mrae', **common), hyper_none
    #yield 'esldmraeptr', EEMQ(newLR(), optim='mrae', policy='ptr', **common), hyper_none
    #yield 'esldmraemrae', EEMQ(newLR(), optim='mrae', policy='mrae', **common), hyper_none


def evaluate_experiment(true_prevalences, estim_prevalences):
    print('\nEvaluation Metrics:\n'+'='*22)
    for eval_measure in [qp.error.mae, qp.error.mrae]:
        err = eval_measure(true_prevalences, estim_prevalences)
        print(f'\t{eval_measure.__name__}={err:.4f}')
    print()


def evaluate_method_point_test(true_prev, estim_prev):
    print('\nPoint-Test evaluation:\n' + '=' * 22)
    print(f'true-prev={F.strprev(true_prev)}, estim-prev={F.strprev(estim_prev)}')
    for eval_measure in [qp.error.mae, qp.error.mrae]:
        err = eval_measure(true_prev, estim_prev)
        print(f'\t{eval_measure.__name__}={err:.4f}')


def result_path(path, dataset_name, model_name, optim_loss):
    return os.path.join(path, f'{dataset_name}-{model_name}-{optim_loss}.pkl')


def is_already_computed(dataset_name, model_name, optim_loss):
    if dataset_name=='semeval':
        check_datasets = ['semeval13', 'semeval14', 'semeval15']
    else:
        check_datasets = [dataset_name]
    return all(os.path.exists(result_path(args.results, name, model_name, optim_loss)) for name in check_datasets)


def save_results(dataset_name, model_name, optim_loss, *results):
    rpath = result_path(args.results, dataset_name, model_name, optim_loss)
    qp.util.create_parent_dir(rpath)
    with open(rpath, 'wb') as foo:
        pickle.dump(tuple(results), foo, pickle.HIGHEST_PROTOCOL)


def run(experiment):

    optim_loss, dataset_name, (model_name, model, hyperparams) = experiment

    if is_already_computed(dataset_name, model_name, optim_loss=optim_loss):
        print(f'result for dataset={dataset_name} model={model_name} loss={optim_loss} already computed.')
        return
    elif (optim_loss == 'mae' and 'mrae' in model_name) or (optim_loss=='mrae' and 'mae' in model_name):
        print(f'skipping model={model_name} for optim_loss={optim_loss}')
        return
    else:
        print(f'running dataset={dataset_name} model={model_name} loss={optim_loss}')

    benchmark_devel = qp.datasets.fetch_twitter(dataset_name, for_model_selection=True, min_df=5, pickle=True)
    benchmark_devel.stats()

    # model selection (hyperparameter optimization for a quantification-oriented loss)
    if hyperparams is not None:
        model_selection = qp.model_selection.GridSearchQ(
            model,
            param_grid=hyperparams,
            sample_size=settings.SAMPLE_SIZE,
            n_prevpoints=21,
            n_repetitions=5,
            error=optim_loss,
            refit=False,
            timeout=60*60,
            verbose=True
        )
        model_selection.fit(benchmark_devel.training, benchmark_devel.test)
        model = model_selection.best_model()
        best_params = model_selection.best_params_
    else:
        best_params = {}

    # model evaluation
    test_names = [dataset_name] if dataset_name != 'semeval' else ['semeval13', 'semeval14', 'semeval15']
    for test_no, test_name in enumerate(test_names):
        benchmark_eval = qp.datasets.fetch_twitter(test_name, for_model_selection=False, min_df=5, pickle=True)
        if test_no == 0:
            print('fitting the selected model')
            # fits the model only the first time
            model.fit(benchmark_eval.training)

        true_prevalences, estim_prevalences = qp.evaluation.artificial_sampling_prediction(
            model,
            test=benchmark_eval.test,
            sample_size=settings.SAMPLE_SIZE,
            n_prevpoints=21,
            n_repetitions=25,
            n_jobs=-1 if isinstance(model, qp.method.meta.Ensemble) else 1
        )
        test_estim_prevalence = model.quantify(benchmark_eval.test.instances)
        test_true_prevalence = benchmark_eval.test.prevalence()

        evaluate_experiment(true_prevalences, estim_prevalences)
        evaluate_method_point_test(test_true_prevalence, test_estim_prevalence)
        save_results(test_name, model_name, optim_loss,
                     true_prevalences, estim_prevalences,
                     benchmark_eval.training.prevalence(), test_true_prevalence, test_estim_prevalence,
                     best_params)

    #if isinstance(model, QuaNet):
        #model.clean_checkpoint_dir()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments for Tweeter Sentiment Quantification')
    parser.add_argument('results', metavar='RESULT_PATH', type=str,
                        help='path to the directory where to store the results')
    parser.add_argument('--svmperfpath', metavar='SVMPERF_PATH', type=str, default='./svm_perf_quantification',
                        help='path to the directory with svmperf')
    parser.add_argument('--checkpointdir', metavar='PATH', type=str, default='./checkpoint',
                        help='path to the directory where to dump QuaNet checkpoints')
    args = parser.parse_args()

    print(f'Result folder: {args.results}')
    np.random.seed(0)

    optim_losses = ['mae', 'mrae']
    datasets = qp.datasets.TWITTER_SENTIMENT_DATASETS_TRAIN

    models = quantification_models()
    qp.util.parallel(run, itertools.product(optim_losses, datasets, models), n_jobs=settings.N_JOBS)

    models = quantification_cuda_models()
    qp.util.parallel(run, itertools.product(optim_losses, datasets, models), n_jobs=settings.CUDA_N_JOBS)

    models = quantification_ensembles()
    qp.util.parallel(run, itertools.product(optim_losses, datasets, models), n_jobs=1)
    # Parallel(n_jobs=1)(
    #     delayed(run)(experiment) for experiment in itertools.product(optim_losses, datasets, models)
    # )

    #shutil.rmtree(args.checkpointdir, ignore_errors=True)


