from typing import Union, Callable, Iterable
import numpy as np
from tqdm import tqdm
import inspect
import quapy as qp
from quapy.protocol import AbstractProtocol, OnLabelledCollectionProtocol
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier
from quapy.util import temp_seed
import quapy.functional as F
import pandas as pd


def prediction(model: BaseQuantifier, protocol: AbstractProtocol, verbose=False):
    sout = lambda x: print(x) if verbose else None
    from method.aggregative import AggregativeQuantifier
    if isinstance(model, AggregativeQuantifier) and isinstance(protocol, OnLabelledCollectionProtocol):
        sout('speeding up the prediction for the aggregative quantifier')
        pre_classified = model.classify(protocol.get_labelled_collection().instances)
        return __prediction_helper(model.aggregate, protocol.on_preclassified_instances(pre_classified), verbose)
    else:
        sout(f'the method is not aggregative, or the protocol is not an instance of '
             f'{OnLabelledCollectionProtocol.__name__}, so no optimization can be carried out')
        return __prediction_helper(model.quantify, protocol, verbose)


def __prediction_helper(quantification_fn, protocol: AbstractProtocol, verbose=False):
    true_prevs, estim_prevs = [], []
    for sample in tqdm(protocol(), total=protocol.total()) if verbose else protocol():
        estim_prevs.append(quantification_fn(sample.instances))
        true_prevs.append(sample.prevalence())

    true_prevs = np.asarray(true_prevs)
    estim_prevs = np.asarray(estim_prevs)

    return true_prevs, estim_prevs


def evaluation_report(model: BaseQuantifier,
                      protocol: AbstractProtocol,
                      error_metrics:Iterable[Union[str,Callable]]='mae',
                      verbose=False):

    true_prevs, estim_prevs = prediction(model, protocol, verbose)
    return _prevalence_report(true_prevs, estim_prevs, error_metrics)


def _prevalence_report(true_prevs, estim_prevs, error_metrics: Iterable[Union[str, Callable]] = 'mae'):

    if isinstance(error_metrics, str):
        error_metrics = [error_metrics]

    error_funcs = [qp.error.from_name(e) if isinstance(e, str) else e for e in error_metrics]
    assert all(hasattr(e, '__call__') for e in error_funcs), 'invalid error functions'
    error_names = [e.__name__ for e in error_funcs]

    df = pd.DataFrame(columns=['true-prev', 'estim-prev'] + error_names)
    for true_prev, estim_prev in zip(true_prevs, estim_prevs):
        series = {'true-prev': true_prev, 'estim-prev': estim_prev}
        for error_name, error_metric in zip(error_names, error_funcs):
            score = error_metric(true_prev, estim_prev)
            series[error_name] = score
        df = df.append(series, ignore_index=True)

    return df


def evaluate(model: BaseQuantifier, protocol: AbstractProtocol, error_metric:Union[str, Callable], verbose=False):
    if isinstance(error_metric, str):
        error_metric = qp.error.from_name(error_metric)
    true_prevs, estim_prevs = prediction(model, protocol, verbose)
    return error_metric(true_prevs, estim_prevs)



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

