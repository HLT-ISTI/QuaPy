from typing import Union, Callable, Iterable
import numpy as np
from tqdm import tqdm
import quapy as qp
from quapy.protocol import AbstractProtocol, OnLabelledCollectionProtocol, IterateProtocol
from quapy.method.base import BaseQuantifier
import pandas as pd


def prediction(
        model: BaseQuantifier,
        protocol: AbstractProtocol,
        aggr_speedup: Union[str, bool] = 'auto',
        verbose=False):
    """
    Uses a quantification model to generate predictions for the samples generated via a specific protocol.
    This function is central to all evaluation processes, and is endowed with an optimization to speed-up the
    prediction of protocols that generate samples from a large collection. The optimization applies to aggregative
    quantifiers only, and to OnLabelledCollectionProtocol protocols, and comes down to generating the classification
    predictions once and for all, and then generating samples over the classification predictions (instead of over
    the raw instances), so that the classifier prediction is never called again. This behaviour is obtained by
    setting `aggr_speedup` to 'auto' or True, and is only carried out if the overall process is convenient in terms
    of computations (e.g., if the number of classification predictions needed for the original collection exceed the
    number of classification predictions needed for all samples, then the optimization is not undertaken).

    :param model: a quantifier, instance of :class:`quapy.method.base.BaseQuantifier`
    :param protocol: :class:`quapy.protocol.AbstractProtocol`; if this object is also instance of
        :class:`quapy.protocol.OnLabelledCollectionProtocol`, then the aggregation speed-up can be run. This is the protocol
        in charge of generating the samples for which the model has to issue class prevalence predictions.
    :param aggr_speedup: whether or not to apply the speed-up. Set to "force" for applying it even if the number of
        instances in the original collection on which the protocol acts is larger than the number of instances
        in the samples to be generated. Set to True or "auto" (default) for letting QuaPy decide whether it is
        convenient or not. Set to False to deactivate.
    :param verbose: boolean, show or not information in stdout
    :return: a tuple `(true_prevs, estim_prevs)` in which each element in the tuple is an array of shape
        `(n_samples, n_classes)` containing the true, or predicted, prevalence values for each sample
    """
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
    for sample_instances, sample_prev in tqdm(protocol(), total=protocol.total(), desc='predicting') if verbose else protocol():
        estim_prevs.append(quantification_fn(sample_instances))
        true_prevs.append(sample_prev)

    true_prevs = np.asarray(true_prevs)
    estim_prevs = np.asarray(estim_prevs)

    return true_prevs, estim_prevs


def evaluation_report(model: BaseQuantifier,
                      protocol: AbstractProtocol,
                      error_metrics: Iterable[Union[str,Callable]] = 'mae',
                      aggr_speedup: Union[str, bool] = 'auto',
                      verbose=False):
    """
    Generates a report (a pandas' DataFrame) containing information of the evaluation of the model as according
    to a specific protocol and in terms of one or more evaluation metrics (errors).


    :param model: a quantifier, instance of :class:`quapy.method.base.BaseQuantifier`
    :param protocol: :class:`quapy.protocol.AbstractProtocol`; if this object is also instance of
        :class:`quapy.protocol.OnLabelledCollectionProtocol`, then the aggregation speed-up can be run. This is the protocol
        in charge of generating the samples in which the model is evaluated.
    :param error_metrics: a string, or list of strings, representing the name(s) of an error function in `qp.error`
        (e.g., 'mae', the default value), or a callable function, or a list of callable functions, implementing
        the error function itself.
    :param aggr_speedup: whether or not to apply the speed-up. Set to "force" for applying it even if the number of
        instances in the original collection on which the protocol acts is larger than the number of instances
        in the samples to be generated. Set to True or "auto" (default) for letting QuaPy decide whether it is
        convenient or not. Set to False to deactivate.
    :param verbose: boolean, show or not information in stdout
    :return: a pandas' DataFrame containing the columns 'true-prev' (the true prevalence of each sample),
        'estim-prev' (the prevalence estimated by the model for each sample), and as many columns as error metrics
        have been indicated, each displaying the score in terms of that metric for every sample.
    """

    true_prevs, estim_prevs = prediction(model, protocol, aggr_speedup=aggr_speedup, verbose=verbose)
    return _prevalence_report(true_prevs, estim_prevs, error_metrics)


def _prevalence_report(true_prevs, estim_prevs, error_metrics: Iterable[Union[str, Callable]] = 'mae'):

    if isinstance(error_metrics, str):
        error_metrics = [error_metrics]

    error_funcs = [qp.error.from_name(e) if isinstance(e, str) else e for e in error_metrics]
    assert all(hasattr(e, '__call__') for e in error_funcs), 'invalid error functions'
    error_names = [e.__name__ for e in error_funcs]

    row_entries = []
    for true_prev, estim_prev in zip(true_prevs, estim_prevs):
        series = {'true-prev': true_prev, 'estim-prev': estim_prev}
        for error_name, error_metric in zip(error_names, error_funcs):
            score = error_metric(true_prev, estim_prev)
            series[error_name] = score
        row_entries.append(series)

    df = pd.DataFrame.from_records(row_entries)
    return df


def evaluate(
        model: BaseQuantifier,
        protocol: AbstractProtocol,
        error_metric: Union[str, Callable],
        aggr_speedup: Union[str, bool] = 'auto',
        verbose=False):
    """
    Evaluates a quantification model according to a specific sample generation protocol and in terms of one
    evaluation metric (error).

    :param model: a quantifier, instance of :class:`quapy.method.base.BaseQuantifier`
    :param protocol: :class:`quapy.protocol.AbstractProtocol`; if this object is also instance of
        :class:`quapy.protocol.OnLabelledCollectionProtocol`, then the aggregation speed-up can be run. This is the
        protocol in charge of generating the samples in which the model is evaluated.
    :param error_metric: a string representing the name(s) of an error function in `qp.error`
        (e.g., 'mae'), or a callable function implementing the error function itself.
    :param aggr_speedup: whether or not to apply the speed-up. Set to "force" for applying it even if the number of
        instances in the original collection on which the protocol acts is larger than the number of instances
        in the samples to be generated. Set to True or "auto" (default) for letting QuaPy decide whether it is
        convenient or not. Set to False to deactivate.
    :param verbose: boolean, show or not information in stdout
    :return: if the error metric is not averaged (e.g., 'ae', 'rae'), returns an array of shape `(n_samples,)` with
        the error scores for each sample; if the error metric is averaged (e.g., 'mae', 'mrae') then returns
        a single float
    """

    if isinstance(error_metric, str):
        error_metric = qp.error.from_name(error_metric)
    true_prevs, estim_prevs = prediction(model, protocol, aggr_speedup=aggr_speedup, verbose=verbose)
    return error_metric(true_prevs, estim_prevs)


def evaluate_on_samples(
        model: BaseQuantifier,
        samples: Iterable[qp.data.LabelledCollection],
        error_metric: Union[str, Callable],
        verbose=False):
    """
    Evaluates a quantification model on a given set of samples and in terms of one evaluation metric (error).

    :param model: a quantifier, instance of :class:`quapy.method.base.BaseQuantifier`
    :param samples: a list of samples on which the quantifier is to be evaluated
    :param error_metric: a string representing the name(s) of an error function in `qp.error`
        (e.g., 'mae'), or a callable function implementing the error function itself.
    :param verbose: boolean, show or not information in stdout
    :return: if the error metric is not averaged (e.g., 'ae', 'rae'), returns an array of shape `(n_samples,)` with
        the error scores for each sample; if the error metric is averaged (e.g., 'mae', 'mrae') then returns
        a single float
    """

    return evaluate(model, IterateProtocol(samples), error_metric, aggr_speedup=False, verbose=verbose)





