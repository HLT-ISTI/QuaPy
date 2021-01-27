from typing import Union, Callable, Iterable

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier
from quapy.util import temp_seed
import quapy.functional as F


def artificial_sampling_prediction(
        model: BaseQuantifier,
        test: LabelledCollection,
        sample_size,
        n_prevpoints=210,
        n_repetitions=1,
        n_jobs=1,
        random_seed=42,
        verbose=True
):
    """
    Performs the predictions for all samples generated according to the artificial sampling protocol.
    :param model: the model in charge of generating the class prevalence estimations
    :param test: the test set on which to perform arificial sampling
    :param sample_size: the size of the samples
    :param n_prevpoints: the number of different prevalences to sample
    :param n_repetitions: the number of repetitions for each prevalence
    :param n_jobs: number of jobs to be run in parallel
    :param random_seed: allows to replicate the samplings. The seed is local to the method and does not affect
    any other random process.
    :param verbose: if True, shows a progress bar
    :return: two ndarrays of shape (m,n) with m the number of samples (n_prevpoints*n_repetitions) and n the
     number of classes. The first one contains the true prevalences for the samples generated while the second one
     contains the the prevalence estimations
    """

    with temp_seed(random_seed):
        indexes = list(test.artificial_sampling_index_generator(sample_size, n_prevpoints, n_repetitions))

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

    pbar = tqdm(indexes, desc='[artificial sampling protocol] predicting') if verbose else indexes
    results = qp.util.parallel(_predict_prevalences, pbar, n_jobs=n_jobs)
    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(_predict_prevalences)(index) for index in pbar
    # )

    true_prevalences, estim_prevalences = zip(*results)
    true_prevalences = np.asarray(true_prevalences)
    estim_prevalences = np.asarray(estim_prevalences)

    return true_prevalences, estim_prevalences


def evaluate(model: BaseQuantifier, test_samples:Iterable[LabelledCollection], err:Union[str, Callable], n_jobs:int=-1):
    if isinstance(err, str):
        err = qp.error.from_name(err)
    scores = qp.util.parallel(_delayed_eval, ((model, Ti, err) for Ti in test_samples), n_jobs=n_jobs)
    # scores = Parallel(n_jobs=n_jobs)(
    #     delayed(_delayed_eval)(model, Ti, err) for Ti in test_samples
    # )
    return np.mean(scores)


def _delayed_eval(args):
    model, test, error = args
    prev_estim = model.quantify(test.instances)
    prev_true  = test.prevalence()
    return error(prev_true, prev_estim)

