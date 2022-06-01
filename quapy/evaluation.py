from typing import Union, Callable, Iterable
import numpy as np
from tqdm import tqdm
import quapy as qp
from quapy.protocol import AbstractProtocol, OnLabelledCollectionProtocol
from quapy.method.base import BaseQuantifier
import pandas as pd


def prediction(model: BaseQuantifier, protocol: AbstractProtocol, aggr_speedup='auto', verbose=False):
    assert aggr_speedup in [False, True, 'auto', 'force'], 'invalid value for aggr_speedup'

    sout = lambda x: print(x) if verbose else None

    apply_optimization = False

    if aggr_speedup in [True, 'auto', 'force']:
        # checks whether the prediction can be made more efficiently; this check consists in verifying if the model is
        # of type aggregative, if the protocol is based on LabelledCollection, and if the total number of documents to
        # classify using the protocol would exceed the number of test documents in the original collection
        from quapy.method.aggregative import AggregativeQuantifier
        if isinstance(model, AggregativeQuantifier) and isinstance(protocol, OnLabelledCollectionProtocol):
            if aggr_speedup == 'force':
                apply_optimization = True
                sout(f'forcing aggregative speedup')
            elif hasattr(protocol, 'sample_size'):
                nD = len(protocol.get_labelled_collection())
                samplesD = protocol.total() * protocol.sample_size
                if nD < samplesD:
                    apply_optimization = True
                    sout(f'speeding up the prediction for the aggregative quantifier, '
                         f'total classifications {nD} instead of {samplesD}')

    if apply_optimization:
        pre_classified = model.classify(protocol.get_labelled_collection().instances)
        protocol_with_predictions = protocol.on_preclassified_instances(pre_classified)
        return __prediction_helper(model.aggregate, protocol_with_predictions, verbose)
    else:
        return __prediction_helper(model.quantify, protocol, verbose)


def __prediction_helper(quantification_fn, protocol: AbstractProtocol, verbose=False):
    true_prevs, estim_prevs = [], []
    for sample_instances, sample_prev in tqdm(protocol(), total=protocol.total()) if verbose else protocol():
        estim_prevs.append(quantification_fn(sample_instances))
        true_prevs.append(sample_prev)

    true_prevs = np.asarray(true_prevs)
    estim_prevs = np.asarray(estim_prevs)

    return true_prevs, estim_prevs


def evaluation_report(model: BaseQuantifier,
                      protocol: AbstractProtocol,
                      error_metrics: Iterable[Union[str,Callable]] = 'mae',
                      aggr_speedup='auto',
                      verbose=False):

    true_prevs, estim_prevs = prediction(model, protocol, aggr_speedup=aggr_speedup, verbose=verbose)
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


def evaluate(
        model: BaseQuantifier,
        protocol: AbstractProtocol,
        error_metric:Union[str, Callable],
        aggr_speedup='auto',
        verbose=False):

    if isinstance(error_metric, str):
        error_metric = qp.error.from_name(error_metric)
    true_prevs, estim_prevs = prediction(model, protocol, aggr_speedup=aggr_speedup, verbose=verbose)
    return error_metric(true_prevs, estim_prevs)




