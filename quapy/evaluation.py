from typing import Union, Callable, Iterable
import numpy as np
from tqdm import tqdm
import inspect

import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier
from quapy.util import temp_seed, _check_sample_size
import quapy.functional as F
import pandas as pd


def artificial_prevalence_prediction(
        model: BaseQuantifier,
        test: LabelledCollection,
        sample_size=None,
        n_prevpoints=101,
        n_repetitions=1,
        eval_budget: int = None,
        n_jobs=1,
        random_seed=42,
        verbose=False):
    """
    Performs the predictions for all samples generated according to the Artificial Prevalence Protocol (APP).
    The APP consists of exploring a grid of prevalence values containing `n_prevalences` points (e.g.,
    [0, 0.05, 0.1, 0.15, ..., 1], if `n_prevalences=21`), and generating all valid combinations of
    prevalence values for all classes (e.g., for 3 classes, samples with [0, 0, 1], [0, 0.05, 0.95], ...,
    [1, 0, 0] prevalence values of size `sample_size` will be considered). The number of samples for each valid
    combination of prevalence values is indicated by `repeats`.

    :param model: the model in charge of generating the class prevalence estimations
    :param test: the test set on which to perform APP
    :param sample_size: integer, the size of the samples; if None, then the sample size is
        taken from qp.environ['SAMPLE_SIZE']
    :param n_prevpoints: integer, the number of different prevalences to sample (or set to None if eval_budget
        is specified; default 101, i.e., steps of 1%)
    :param n_repetitions: integer, the number of repetitions for each prevalence (default 1)
    :param eval_budget: integer, if specified, sets a ceil on the number of evaluations to perform. For example, if
        there are 3 classes, `repeats=1`, and `eval_budget=20`, then `n_prevpoints` will be set to 5, since this
        will generate 15 different prevalence vectors ([0, 0, 1], [0, 0.25, 0.75], [0, 0.5, 0.5] ... [1, 0, 0]) and
        since setting `n_prevpoints=6` would produce more than 20 evaluations.
    :param n_jobs: integer, number of jobs to be run in parallel (default 1)
    :param random_seed: integer, allows to replicate the samplings. The seed is local to the method and does not affect
        any other random process (default 42)
    :param verbose: if True, shows a progress bar
    :return: a tuple containing two `np.ndarrays` of shape `(m,n,)` with `m` the number of samples
        `(n_prevpoints*repeats)` and `n` the number of classes. The first one contains the true prevalence values
        for the samples generated while the second one contains the prevalence estimations
    """

    sample_size = _check_sample_size(sample_size)
    n_prevpoints, _ = qp.evaluation._check_num_evals(test.n_classes, n_prevpoints, eval_budget, n_repetitions, verbose)

    with temp_seed(random_seed):
        indexes = list(test.artificial_sampling_index_generator(sample_size, n_prevpoints, n_repetitions))

    return _predict_from_indexes(indexes, model, test, n_jobs, verbose)


def natural_prevalence_prediction(
        model: BaseQuantifier,
        test: LabelledCollection,
        sample_size=None,
        repeats=100,
        n_jobs=1,
        random_seed=42,
        verbose=False):
    """
    Performs the predictions for all samples generated according to the Natural Prevalence Protocol (NPP).
    The NPP consists of drawing samples uniformly at random, therefore approximately preserving the natural
    prevalence of the collection.

    :param model: the model in charge of generating the class prevalence estimations
    :param test: the test set on which to perform NPP
    :param sample_size: integer, the size of the samples; if None, then the sample size is
        taken from qp.environ['SAMPLE_SIZE']
    :param repeats: integer, the number of samples to generate (default 100)
    :param n_jobs: integer, number of jobs to be run in parallel (default 1)
    :param random_seed: allows to replicate the samplings. The seed is local to the method and does not affect
        any other random process (default 42)
    :param verbose: if True, shows a progress bar
    :return: a tuple containing two `np.ndarrays` of shape `(m,n,)` with `m` the number of samples
        `(repeats)` and `n` the number of classes. The first one contains the true prevalence values
        for the samples generated while the second one contains the prevalence estimations
    """

    sample_size = _check_sample_size(sample_size)
    with temp_seed(random_seed):
        indexes = list(test.natural_sampling_index_generator(sample_size, repeats))

    return _predict_from_indexes(indexes, model, test, n_jobs, verbose)


def gen_prevalence_prediction(model: BaseQuantifier, gen_fn: Callable, eval_budget=None):
    """
    Generates prevalence predictions for a custom protocol defined as a generator function that yields
    samples at each iteration. The sequence of samples is processed exhaustively if `eval_budget=None`
    or up to the `eval_budget` iterations if specified.

    :param model: the model in charge of generating the class prevalence estimations
    :param gen_fn: a generator function yielding one sample at each iteration
    :param eval_budget: a maximum number of evaluations to run. Set to None (default) for exploring the
        entire sequence
    :return: a tuple containing two `np.ndarrays` of shape `(m,n,)` with `m` the number of samples
        generated and `n` the number of classes. The first one contains the true prevalence values
        for the samples generated while the second one contains the prevalence estimations
    """
    if not inspect.isgenerator(gen_fn()):
        raise ValueError('param "gen_fun" is not a callable returning a generator')

    if not isinstance(eval_budget, int):
        eval_budget = -1

    true_prevalences, estim_prevalences = [], []
    for sample_instances, true_prev in gen_fn():
        true_prevalences.append(true_prev)
        estim_prevalences.append(model.quantify(sample_instances))
        eval_budget -= 1
        if eval_budget == 0:
            break

    true_prevalences = np.asarray(true_prevalences)
    estim_prevalences = np.asarray(estim_prevalences)

    return true_prevalences, estim_prevalences


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


def artificial_prevalence_report(
        model: BaseQuantifier,
        test: LabelledCollection,
        sample_size=None,
        n_prevpoints=101,
        n_repetitions=1,
        eval_budget: int = None,
        n_jobs=1,
        random_seed=42,
        error_metrics:Iterable[Union[str,Callable]]='mae',
        verbose=False):
    """
    Generates an evaluation report for all samples generated according to the Artificial Prevalence Protocol (APP).
    The APP consists of exploring a grid of prevalence values containing `n_prevalences` points (e.g.,
    [0, 0.05, 0.1, 0.15, ..., 1], if `n_prevalences=21`), and generating all valid combinations of
    prevalence values for all classes (e.g., for 3 classes, samples with [0, 0, 1], [0, 0.05, 0.95], ...,
    [1, 0, 0] prevalence values of size `sample_size` will be considered). The number of samples for each valid
    combination of prevalence values is indicated by `repeats`.
    Te report takes the form of a
    pandas' `dataframe <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_
    in which the rows correspond to different samples, and the columns inform of the true prevalence values,
    the estimated prevalence values, and the score obtained by each of the evaluation measures indicated.

    :param model: the model in charge of generating the class prevalence estimations
    :param test: the test set on which to perform APP
    :param sample_size: integer, the size of the samples; if None, then the sample size is
        taken from qp.environ['SAMPLE_SIZE']
    :param n_prevpoints: integer, the number of different prevalences to sample (or set to None if eval_budget
        is specified; default 101, i.e., steps of 1%)
    :param n_repetitions: integer, the number of repetitions for each prevalence (default 1)
    :param eval_budget: integer, if specified, sets a ceil on the number of evaluations to perform. For example, if
        there are 3 classes, `repeats=1`, and `eval_budget=20`, then `n_prevpoints` will be set to 5, since this
        will generate 15 different prevalence vectors ([0, 0, 1], [0, 0.25, 0.75], [0, 0.5, 0.5] ... [1, 0, 0]) and
        since setting `n_prevpoints=6` would produce more than 20 evaluations.
    :param n_jobs: integer, number of jobs to be run in parallel (default 1)
    :param random_seed: integer, allows to replicate the samplings. The seed is local to the method and does not affect
        any other random process (default 42)
    :param error_metrics: a string indicating the name of the error (as defined in :mod:`quapy.error`) or a
        callable error function; optionally, a list of strings or callables can be indicated, if the results
        are to be evaluated with more than one error metric. Default is "mae"
    :param verbose: if True, shows a progress bar
    :return: pandas' dataframe with rows corresponding to different samples, and with columns informing of the
        true prevalence values, the estimated prevalence values, and the score obtained by each of the evaluation
        measures indicated.
    """

    true_prevs, estim_prevs = artificial_prevalence_prediction(
        model, test, sample_size, n_prevpoints, n_repetitions, eval_budget, n_jobs, random_seed, verbose
    )
    return _prevalence_report(true_prevs, estim_prevs, error_metrics)


def natural_prevalence_report(
        model: BaseQuantifier,
        test: LabelledCollection,
        sample_size=None,
        repeats=100,
        n_jobs=1,
        random_seed=42,
        error_metrics:Iterable[Union[str,Callable]]='mae',
        verbose=False):
    """
    Generates an evaluation report for all samples generated according to the Natural Prevalence Protocol (NPP).
    The NPP consists of drawing samples uniformly at random, therefore approximately preserving the natural
    prevalence of the collection.
    Te report takes the form of a
    pandas' `dataframe <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_
    in which the rows correspond to different samples, and the columns inform of the true prevalence values,
    the estimated prevalence values, and the score obtained by each of the evaluation measures indicated.

    :param model: the model in charge of generating the class prevalence estimations
    :param test: the test set on which to perform NPP
    :param sample_size: integer, the size of the samples; if None, then the sample size is
        taken from qp.environ['SAMPLE_SIZE']
    :param repeats: integer, the number of samples to generate (default 100)
    :param n_jobs: integer, number of jobs to be run in parallel (default 1)
    :param random_seed: allows to replicate the samplings. The seed is local to the method and does not affect
        any other random process (default 42)
    :param error_metrics: a string indicating the name of the error (as defined in :mod:`quapy.error`) or a
        callable error function; optionally, a list of strings or callables can be indicated, if the results
        are to be evaluated with more than one error metric. Default is "mae"
    :param verbose: if True, shows a progress bar
    :return: a tuple containing two `np.ndarrays` of shape `(m,n,)` with `m` the number of samples
        `(repeats)` and `n` the number of classes. The first one contains the true prevalence values
        for the samples generated while the second one contains the prevalence estimations

    """
    sample_size = _check_sample_size(sample_size)
    true_prevs, estim_prevs = natural_prevalence_prediction(
        model, test, sample_size, repeats, n_jobs, random_seed, verbose
    )
    return _prevalence_report(true_prevs, estim_prevs, error_metrics)


def gen_prevalence_report(model: BaseQuantifier, gen_fn: Callable, eval_budget=None,
                          error_metrics:Iterable[Union[str,Callable]]='mae'):
    """
    GGenerates an evaluation report for a custom protocol defined as a generator function that yields
    samples at each iteration. The sequence of samples is processed exhaustively if `eval_budget=None`
    or up to the `eval_budget` iterations if specified.
    Te report takes the form of a
    pandas' `dataframe <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_
    in which the rows correspond to different samples, and the columns inform of the true prevalence values,
    the estimated prevalence values, and the score obtained by each of the evaluation measures indicated.

    :param model: the model in charge of generating the class prevalence estimations
    :param gen_fn: a generator function yielding one sample at each iteration
    :param eval_budget: a maximum number of evaluations to run. Set to None (default) for exploring the
        entire sequence
    :return: a tuple containing two `np.ndarrays` of shape `(m,n,)` with `m` the number of samples
        generated. The first one contains the true prevalence values
        for the samples generated while the second one contains the prevalence estimations
    """
    true_prevs, estim_prevs = gen_prevalence_prediction(model, gen_fn, eval_budget)
    return _prevalence_report(true_prevs, estim_prevs, error_metrics)


def _prevalence_report(
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


def artificial_prevalence_protocol(
        model: BaseQuantifier,
        test: LabelledCollection,
        sample_size=None,
        n_prevpoints=101,
        repeats=1,
        eval_budget: int = None,
        n_jobs=1,
        random_seed=42,
        error_metric:Union[str,Callable]='mae',
        verbose=False):
    """
    Generates samples according to the Artificial Prevalence Protocol (APP).
    The APP consists of exploring a grid of prevalence values containing `n_prevalences` points (e.g.,
    [0, 0.05, 0.1, 0.15, ..., 1], if `n_prevalences=21`), and generating all valid combinations of
    prevalence values for all classes (e.g., for 3 classes, samples with [0, 0, 1], [0, 0.05, 0.95], ...,
    [1, 0, 0] prevalence values of size `sample_size` will be considered). The number of samples for each valid
    combination of prevalence values is indicated by `repeats`.

    :param model: the model in charge of generating the class prevalence estimations
    :param test: the test set on which to perform APP
    :param sample_size: integer, the size of the samples; if None, then the sample size is
        taken from qp.environ['SAMPLE_SIZE']
    :param n_prevpoints: integer, the number of different prevalences to sample (or set to None if eval_budget
        is specified; default 101, i.e., steps of 1%)
    :param repeats: integer, the number of repetitions for each prevalence (default 1)
    :param eval_budget: integer, if specified, sets a ceil on the number of evaluations to perform. For example, if
        there are 3 classes, `repeats=1`, and `eval_budget=20`, then `n_prevpoints` will be set to 5, since this
        will generate 15 different prevalence vectors ([0, 0, 1], [0, 0.25, 0.75], [0, 0.5, 0.5] ... [1, 0, 0]) and
        since setting `n_prevpoints=6` would produce more than 20 evaluations.
    :param n_jobs: integer, number of jobs to be run in parallel (default 1)
    :param random_seed: integer, allows to replicate the samplings. The seed is local to the method and does not affect
        any other random process (default 42)
    :param error_metric: a string indicating the name of the error (as defined in :mod:`quapy.error`) or a
        callable error function
    :param verbose: set to True (default False) for displaying some information on standard output
    :return: yields one sample at a time
    """

    if isinstance(error_metric, str):
        error_metric = qp.error.from_name(error_metric)

    assert hasattr(error_metric, '__call__'), 'invalid error function'

    true_prevs, estim_prevs = artificial_prevalence_prediction(
        model, test, sample_size, n_prevpoints, repeats, eval_budget, n_jobs, random_seed, verbose
    )

    return error_metric(true_prevs, estim_prevs)


def natural_prevalence_protocol(
        model: BaseQuantifier,
        test: LabelledCollection,
        sample_size=None,
        repeats=100,
        n_jobs=1,
        random_seed=42,
        error_metric:Union[str,Callable]='mae',
        verbose=False):
    """
    Generates samples according to the Natural Prevalence Protocol (NPP).
    The NPP consists of drawing samples uniformly at random, therefore approximately preserving the natural
    prevalence of the collection.

    :param model: the model in charge of generating the class prevalence estimations
    :param test: the test set on which to perform NPP
    :param sample_size: integer, the size of the samples; if None, then the sample size is
        taken from qp.environ['SAMPLE_SIZE']
    :param repeats: integer, the number of samples to generate
    :param n_jobs: integer, number of jobs to be run in parallel (default 1)
    :param random_seed: allows to replicate the samplings. The seed is local to the method and does not affect
        any other random process (default 42)
    :param error_metric: a string indicating the name of the error (as defined in :mod:`quapy.error`) or a
        callable error function
    :param verbose: if True, shows a progress bar
    :return: yields one sample at a time
    """

    if isinstance(error_metric, str):
        error_metric = qp.error.from_name(error_metric)

    assert hasattr(error_metric, '__call__'), 'invalid error function'

    true_prevs, estim_prevs = natural_prevalence_prediction(
        model, test, sample_size, repeats, n_jobs, random_seed, verbose
    )

    return error_metric(true_prevs, estim_prevs)


def evaluate(model: BaseQuantifier, test_samples:Iterable[LabelledCollection], error_metric:Union[str, Callable], n_jobs:int=-1):
    """
    Evaluates a model on a sequence of test samples in terms of a given error metric.

    :param model: the model in charge of generating the class prevalence estimations
    :param test_samples: an iterable yielding one sample at a time
    :param error_metric: a string indicating the name of the error (as defined in :mod:`quapy.error`) or a
        callable error function
    :param n_jobs: integer, number of jobs to be run in parallel (default 1)
    :return: the score obtained using `error_metric`
    """
    if isinstance(error_metric, str):
        error_metric = qp.error.from_name(error_metric)
    scores = qp.util.parallel(_delayed_eval, ((model, Ti, error_metric) for Ti in test_samples), n_jobs=n_jobs)
    return np.mean(scores)


def _delayed_eval(args):
    model, test, error = args
    prev_estim = model.quantify(test.instances)
    prev_true  = test.prevalence()
    return error(prev_true, prev_estim)


def _check_num_evals(n_classes, n_prevpoints=None, eval_budget=None, repeats=1, verbose=False):
    if n_prevpoints is None and eval_budget is None:
        raise ValueError('either n_prevpoints or eval_budget has to be specified')
    elif n_prevpoints is None:
        assert eval_budget > 0, 'eval_budget must be a positive integer'
        n_prevpoints = F.get_nprevpoints_approximation(eval_budget, n_classes, repeats)
        eval_computations = F.num_prevalence_combinations(n_prevpoints, n_classes, repeats)
        if verbose:
            print(f'setting n_prevpoints={n_prevpoints} so that the number of '
                  f'evaluations ({eval_computations}) does not exceed the evaluation '
                  f'budget ({eval_budget})')
    elif eval_budget is None:
        eval_computations = F.num_prevalence_combinations(n_prevpoints, n_classes, repeats)
        if verbose:
            print(f'{eval_computations} evaluations will be performed for each '
                  f'combination of hyper-parameters')
    else:
        eval_computations = F.num_prevalence_combinations(n_prevpoints, n_classes, repeats)
        if eval_computations > eval_budget:
            n_prevpoints = F.get_nprevpoints_approximation(eval_budget, n_classes, repeats)
            new_eval_computations = F.num_prevalence_combinations(n_prevpoints, n_classes, repeats)
            if verbose:
                print(f'the budget of evaluations would be exceeded with '
                  f'n_prevpoints={n_prevpoints}. Chaning to n_prevpoints={n_prevpoints}. This will produce '
                  f'{new_eval_computations} evaluation computations for each hyper-parameter combination.')
    return n_prevpoints, eval_computations

