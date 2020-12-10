from data import LabelledCollection
from method.base import BaseQuantifier
from utils.util import temp_seed
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def artificial_sampling_prediction(
        model: BaseQuantifier,
        test: LabelledCollection,
        sample_size,
        prevalence_points=21,
        point_repetitions=1,
        n_jobs=-1,
        random_seed=42):
    """
    Performs the predictions for all samples generated according to the artificial sampling protocol.
    :param model: the model in charge of generating the class prevalence estimations
    :param test: the test set on which to perform arificial sampling
    :param sample_size: the size of the samples
    :param prevalence_points: the number of different prevalences to sample
    :param point_repetitions: the number of repetitions for each prevalence
    :param n_jobs: number of jobs to be run in parallel
    :param random_seed: allows to replicate the samplings. The seed is local to the method and does not affect
    any other random process.
    :return: two ndarrays of [m,n] with m the number of samples (prevalence_points*point_repetitions) and n the
     number of classes. The first one contains the true prevalences for the samples generated while the second one
     containing the the prevalences estimations
    """

    with temp_seed(random_seed):
        indexes = list(test.artificial_sampling_index_generator(sample_size, prevalence_points, point_repetitions))

    def _predict_prevalences(index):
        sample = test.sampling_from_index(index)
        true_prevalence = sample.prevalence()
        estim_prevalence = model.quantify(sample.instances)
        return true_prevalence, estim_prevalence

    results = Parallel(n_jobs=n_jobs)(
        delayed(_predict_prevalences)(index) for index in tqdm(indexes)
    )

    true_prevalences, estim_prevalences = zip(*results)
    true_prevalences = np.asarray(true_prevalences)
    estim_prevalences = np.asarray(estim_prevalences)

    return true_prevalences, estim_prevalences




