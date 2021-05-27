from typing import Union, Callable, Iterable

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier
from quapy.util import temp_seed
import quapy.functional as F
import pandas as pd



def artificial_sampling_prediction(
        model: BaseQuantifier,
        test: LabelledCollection,
        sample_size,
        n_prevpoints=210,
        n_repetitions=1,
        eval_budget: int = None,
        n_jobs=1,
        random_seed=42,
        verbose=False):
    """
    Performs the predictions for all samples generated according to the artificial sampling protocol.
    :param model: the model in charge of generating the class prevalence estimations
    :param test: the test set on which to perform arificial sampling
    :param sample_size: the size of the samples
    :param n_prevpoints: the number of different prevalences to sample (or set to None if eval_budget is specified)
    :param n_repetitions: the number of repetitions for each prevalence
    :param eval_budget: if specified, sets a ceil on the number of evaluations to perform. For example, if there are 3
    classes, n_repetitions=1 and eval_budget=20, then n_prevpoints will be set to 5, since this will generate 15
    different prevalences ([0, 0, 1], [0, 0.25, 0.75], [0, 0.5, 0.5] ... [1, 0, 0]) and since setting it n_prevpoints
    to 6 would produce more than 20 evaluations.
    :param n_jobs: number of jobs to be run in parallel
    :param random_seed: allows to replicate the samplings. The seed is local to the method and does not affect
    any other random process.
    :param verbose: if True, shows a progress bar
    :return: two ndarrays of shape (m,n) with m the number of samples (n_prevpoints*n_repetitions) and n the
     number of classes. The first one contains the true prevalences for the samples generated while the second one
     contains the the prevalence estimations
    """

    n_prevpoints, _ = qp.evaluation._check_num_evals(test.n_classes, n_prevpoints, eval_budget, n_repetitions, verbose)

    with temp_seed(random_seed):
        indexes = list(test.artificial_sampling_index_generator(sample_size, n_prevpoints, n_repetitions))

    return _predict_from_indexes(indexes, model, test, n_jobs, verbose)


def natural_sampling_prediction(
        model: BaseQuantifier,
        test: LabelledCollection,
        sample_size,
        n_repetitions=1,
        n_jobs=1,
        random_seed=42,
        verbose=False):
    """
    Performs the predictions for all samples generated according to the artificial sampling protocol.
    :param model: the model in charge of generating the class prevalence estimations
    :param test: the test set on which to perform arificial sampling
    :param sample_size: the size of the samples
    :param n_repetitions: the number of repetitions for each prevalence
    :param n_jobs: number of jobs to be run in parallel
    :param random_seed: allows to replicate the samplings. The seed is local to the method and does not affect
    any other random process.
    :param verbose: if True, shows a progress bar
    :return: two ndarrays of shape (m,n) with m the number of samples (n_repetitions) and n the
     number of classes. The first one contains the true prevalences for the samples generated while the second one
     contains the the prevalence estimations
    """

    with temp_seed(random_seed):
        indexes = list(test.natural_sampling_index_generator(sample_size, n_repetitions))

    return _predict_from_indexes(indexes, model, test, n_jobs, verbose)


def _predict_from_indexes(
        indexes,
        model: BaseQuantifier,
        test: LabelledCollection,
        n_jobs=1,
        verbose=False):

    if model.aggregative: #isinstance(model, qp.method.aggregative.AggregativeQuantifier):
        # print('\tinstance of aggregative-quantifier')
        quantification_func = model.aggregate
        if model.probabilistic: # isinstance(model, qp.method.aggregative.AggregativeProbabilisticQuantifier):
            # print('\t\tinstance of probabilitstic-aggregative-quantifier')
            preclassified_instances = model.posterior_probabilities(test.instances)
        else:
            # print('\t\tinstance of hard-aggregative-quantifier')
            preclassified_instances = model.classify(test.instances)
        test = LabelledCollection(preclassified_instances, test.labels)
    else:
        # print('\t\tinstance of base-quantifier')
        quantification_func = model.quantify

    def _predict_prevalences(index):
        sample = test.sampling_from_index(index)
        true_prevalence = sample.prevalence()
        estim_prevalence = quantification_func(sample.instances)
        return true_prevalence, estim_prevalence

    pbar = tqdm(indexes, desc='[artificial sampling protocol] generating predictions') if verbose else indexes
    results = qp.util.parallel(_predict_prevalences, pbar, n_jobs=n_jobs)

    true_prevalences, estim_prevalences = zip(*results)
    true_prevalences = np.asarray(true_prevalences)
    estim_prevalences = np.asarray(estim_prevalences)

    return true_prevalences, estim_prevalences


def artificial_sampling_report(
        model: BaseQuantifier,
        test: LabelledCollection,
        sample_size,
        n_prevpoints=210,
        n_repetitions=1,
        eval_budget: int = None,
        n_jobs=1,
        random_seed=42,
        error_metrics:Iterable[Union[str,Callable]]='mae',
        verbose=False):

    true_prevs, estim_prevs = artificial_sampling_prediction(
        model, test, sample_size, n_prevpoints, n_repetitions, eval_budget, n_jobs, random_seed, verbose
    )
    return _sampling_report(true_prevs, estim_prevs, error_metrics)


def natural_sampling_report(
        model: BaseQuantifier,
        test: LabelledCollection,
        sample_size,
        n_repetitions=1,
        n_jobs=1,
        random_seed=42,
        error_metrics:Iterable[Union[str,Callable]]='mae',
        verbose=False):

    true_prevs, estim_prevs = natural_sampling_prediction(
        model, test, sample_size, n_repetitions, n_jobs, random_seed, verbose
    )
    return _sampling_report(true_prevs, estim_prevs, error_metrics)


def _sampling_report(
        true_prevs,
        estim_prevs,
        error_metrics: Iterable[Union[str, Callable]] = 'mae'):

    if isinstance(error_metrics, str):
        error_metrics = [error_metrics]

    error_names = [e if isinstance(e, str) else e.__name__ for e in error_metrics]
    error_funcs = [qp.error.from_name(e) if isinstance(e, str) else e for e in error_metrics]
    assert all(hasattr(e, '__call__') for e in error_funcs), 'invalid error functions'

    df = pd.DataFrame(columns=['true-prev', 'estim-prev'] + error_names)
    for true_prev, estim_prev in zip(true_prevs, estim_prevs):
        series = {'true-prev': true_prev, 'estim-prev': estim_prev}
        for error_name, error_metric in zip(error_names, error_funcs):
            score = error_metric(true_prev, estim_prev)
            series[error_name] = score
        df = df.append(series, ignore_index=True)

    return df

def artificial_sampling_eval(
        model: BaseQuantifier,
        test: LabelledCollection,
        sample_size,
        n_prevpoints=210,
        n_repetitions=1,
        eval_budget: int = None,
        n_jobs=1,
        random_seed=42,
        error_metric:Union[str,Callable]='mae',
        verbose=False):

    if isinstance(error_metric, str):
        error_metric = qp.error.from_name(error_metric)

    assert hasattr(error_metric, '__call__'), 'invalid error function'

    true_prevs, estim_prevs = artificial_sampling_prediction(
        model, test, sample_size, n_prevpoints, n_repetitions, eval_budget, n_jobs, random_seed, verbose
    )

    return error_metric(true_prevs, estim_prevs)


def natural_sampling_eval(
        model: BaseQuantifier,
        test: LabelledCollection,
        sample_size,
        n_repetitions=1,
        n_jobs=1,
        random_seed=42,
        error_metric:Union[str,Callable]='mae',
        verbose=False):

    if isinstance(error_metric, str):
        error_metric = qp.error.from_name(error_metric)

    assert hasattr(error_metric, '__call__'), 'invalid error function'

    true_prevs, estim_prevs = natural_sampling_prediction(
        model, test, sample_size, n_repetitions, n_jobs, random_seed, verbose
    )

    return error_metric(true_prevs, estim_prevs)


def evaluate(model: BaseQuantifier, test_samples:Iterable[LabelledCollection], err:Union[str, Callable], n_jobs:int=-1):
    if isinstance(err, str):
        err = qp.error.from_name(err)
    scores = qp.util.parallel(_delayed_eval, ((model, Ti, err) for Ti in test_samples), n_jobs=n_jobs)
    return np.mean(scores)


def _delayed_eval(args):
    model, test, error = args
    prev_estim = model.quantify(test.instances)
    prev_true  = test.prevalence()
    return error(prev_true, prev_estim)


def _check_num_evals(n_classes, n_prevpoints=None, eval_budget=None, n_repetitions=1, verbose=False):
    if n_prevpoints is None and eval_budget is None:
        raise ValueError('either n_prevpoints or eval_budget has to be specified')
    elif n_prevpoints is None:
        assert eval_budget > 0, 'eval_budget must be a positive integer'
        n_prevpoints = F.get_nprevpoints_approximation(eval_budget, n_classes, n_repetitions)
        eval_computations = F.num_prevalence_combinations(n_prevpoints, n_classes, n_repetitions)
        if verbose:
            print(f'setting n_prevpoints={n_prevpoints} so that the number of '
                  f'evaluations ({eval_computations}) does not exceed the evaluation '
                  f'budget ({eval_budget})')
    elif eval_budget is None:
        eval_computations = F.num_prevalence_combinations(n_prevpoints, n_classes, n_repetitions)
        if verbose:
            print(f'{eval_computations} evaluations will be performed for each '
                  f'combination of hyper-parameters')
    else:
        eval_computations = F.num_prevalence_combinations(n_prevpoints, n_classes, n_repetitions)
        if eval_computations > eval_budget:
            n_prevpoints = F.get_nprevpoints_approximation(eval_budget, n_classes, n_repetitions)
            new_eval_computations = F.num_prevalence_combinations(n_prevpoints, n_classes, n_repetitions)
            if verbose:
                print(f'the budget of evaluations would be exceeded with '
                  f'n_prevpoints={n_prevpoints}. Chaning to n_prevpoints={n_prevpoints}. This will produce '
                  f'{new_eval_computations} evaluation computations for each hyper-parameter combination.')
    return n_prevpoints, eval_computations

