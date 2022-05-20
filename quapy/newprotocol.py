import itertools
from collections.abc import Generator
from contextlib import ExitStack
from abc import ABCMeta, abstractmethod

from quapy.data import LabelledCollection
import quapy.functional as F


# 0.1.7
# change the LabelledCollection API (removing protocol-related samplings)
# need to change the two references to the above in the wiki / doc, and code examples...
# removed artificial_prevalence_sampling from functional


# class AbstractProtocol(metaclass=ABCMeta):
#     def __call__(self):
#         for g in self.gen():
#             yield g
#
#     @abstractmethod
#     def gen(self):
#         ...


class AbstractStochasticProtocol(metaclass=ABCMeta):
    def __init__(self, seed=None):
        self.random_seed = seed

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, seed):
        self._random_seed = seed

    @abstractmethod
    def samples_parameters(self):
        """
        This function has to return all the necessary parameters to replicate the samples
        :return: a list of parameters, each of which serves to deterministically generate a sample
        """
        ...

    @abstractmethod
    def sample(self, params):
        """
        Extract one sample determined by the given parameters

        :param params: all the necessary parameters to generate a sample
        :return: one sample (the same sample has to be generated for the same parameters)
        """
        ...

    def __call__(self):
        with ExitStack() as stack:
            if self.random_seed is not None:
                stack.enter_context(qp.util.temp_seed(self.random_seed))
            for params in self.samples_parameters():
                yield self.sample(params)


class APP(AbstractStochasticProtocol):
    """
    Implementation of the artificial prevalence protocol (APP).
    The APP consists of exploring a grid of prevalence values containing `n_prevalences` points (e.g.,
    [0, 0.05, 0.1, 0.15, ..., 1], if `n_prevalences=21`), and generating all valid combinations of
    prevalence values for all classes (e.g., for 3 classes, samples with [0, 0, 1], [0, 0.05, 0.95], ...,
    [1, 0, 0] prevalence values of size `sample_size` will be yielded). The number of samples for each valid
    combination of prevalence values is indicated by `repeats`.

    :param sample_size: integer, number of instances in each sample
    :param n_prevalences: the number of equidistant prevalence points to extract from the [0,1] interval for the
        grid (default is 21)
    :param repeats: number of copies for each valid prevalence vector (default is 1)
    :param random_seed: allows replicating samples across runs (default None)
    """

    def __init__(self, data:LabelledCollection, sample_size, n_prevalences=21, repeats=1, random_seed=None):
        super(APP, self).__init__(random_seed)
        self.data = data
        self.sample_size = sample_size
        self.n_prevalences = n_prevalences
        self.repeats = repeats

    def prevalence_grid(self, dimensions):
        """
        Generates vectors of prevalence values from an exhaustive grid of prevalence values. The
        number of prevalence values explored for each dimension depends on `n_prevalences`, so that, if, for example,
        `n_prevalences=11` then the prevalence values of the grid are taken from [0, 0.1, 0.2, ..., 0.9, 1]. Only
        valid prevalence distributions are returned, i.e., vectors of prevalence values that sum up to 1. For each
        valid vector of prevalence values, `repeat` copies are returned. The vector of prevalence values can be
        implicit (by setting `return_constrained_dim=False`), meaning that the last dimension (which is constrained
        to 1 - sum of the rest) is not returned (note that, quite obviously, in this case the vector does not sum up to
        1). Note that this method is deterministic, i.e., there is no random sampling anywhere.

        :param dimensions: the number of classes
        :return: a `np.ndarray` of shape `(n, dimensions)` if `return_constrained_dim=True` or of shape
            `(n, dimensions-1)` if `return_constrained_dim=False`, where `n` is the number of valid combinations found
            in the grid multiplied by `repeat`
        """
        s = np.linspace(0., 1., self.n_prevalences, endpoint=True)
        s = [s] * (dimensions - 1)
        prevs = [p for p in itertools.product(*s, repeat=1) if sum(p) <= 1]
        prevs = np.asarray(prevs).reshape(len(prevs), -1)
        if self.repeats > 1:
            prevs = np.repeat(prevs, self.repeats, axis=0)
        return prevs

    def samples_parameters(self):
        indexes = []
        for prevs in self.prevalence_grid(dimensions=self.data.n_classes):
            index = data.sampling_index(self.sample_size, *prevs)
            indexes.append(index)
        return indexes

    def sample(self, index):
        return self.data.sampling_from_index(index)


class NPP(AbstractStochasticProtocol):
    """
    A generator of samples that implements the natural prevalence protocol (NPP). The NPP consists of drawing
    samples uniformly at random, therefore approximately preserving the natural prevalence of the collection.

    :param sample_size: integer, the number of instances in each sample
    :param repeats: the number of samples to generate
    """

    def __init__(self, data:LabelledCollection, sample_size, repeats=1, random_seed=None):
        super(NPP, self).__init__(random_seed)
        self.data = data
        self.sample_size = sample_size
        self.repeats = repeats
        self.random_seed = random_seed

    def samples_parameters(self):
        indexes = []
        for _ in range(self.repeats):
            index = data.uniform_sampling_index(self.sample_size)
            indexes.append(index)
        return indexes

    def sample(self, index):
        return self.data.sampling_from_index(index)


class USimplexPP(AbstractStochasticProtocol):

    def __init__(self, data: LabelledCollection, sample_size, repeats=1, random_seed=None):
        super(USimplexPP, self).__init__(random_seed)
        self.data = data
        self.sample_size = sample_size
        self.repeats = repeats
        self.random_seed = random_seed

    def samples_parameters(self):
        indexes = []
        for prevs in F.uniform_simplex_sampling(n_classes=data.n_classes, size=self.repeats):
            index = data.sampling_index(self.sample_size, *prevs)
            indexes.append(index)
        return indexes

    def sample(self, index):
        return self.data.sampling_from_index(index)


class CovariateShift(AbstractStochasticProtocol):
    """
    Generates mixtures of two domains (A and B) at controlled rates, but preserving the original class prevalence.

    :param domainA:
    :param domainB:
    :param sample_size:
    :param repeats:
    :param prevalence: the prevalence to preserv along the mixtures. If specified, should be an array containing
        one prevalence value (positive float) for each class and summing up to one. If not specified, the prevalence
        will be taken from the domain A (default).
    :param mixture_points: an integer indicating the number of points to take from a linear scale (e.g., 21 will
        generate the mixture points [1, 0.95, 0.9, ..., 0]), or the array of mixture values itself.
        the specific points
    :param random_seed:
    """

    def __init__(
            self,
            domainA: LabelledCollection,
            domainB: LabelledCollection,
            sample_size,
            repeats=1,
            prevalence=None,
            mixture_points=11,
            random_seed=None):
        super(CovariateShift, self).__init__(random_seed)
        self.data = data
        self.sample_size = sample_size
        self.repeats = repeats
        if prevalence is None:
            self.prevalence = domainA.prevalence()
        else:
            self.prevalence = np.asarray(prevalence)
            assert len(self.prevalence) == domainA.n_classes, \
                f'wrong shape for the vector prevalence (expected {domainA.n_classes})'
            assert F.check_prevalence_vector(self.prevalence), \
                f'the prevalence vector is not valid (either it contains values outside [0,1] or does not sum up to 1)'
        assert isinstance(mixture_points, int) or 
        self.random_seed = random_seed

    def samples_parameters(self):
        indexes = []
        for _ in range(self.repeats):
            index = data.uniform_sampling_index(self.sample_size)
            indexes.append(index)
        return indexes

    def sample(self, index):
        return self.data.sampling_from_index(index)


if __name__=='__main__':
    import numpy as np
    import quapy as qp

    y = [0]*25 + [1]*25 + [2]*25 + [3]*25
    X = [str(i)+'-'+str(yi) for i, yi in enumerate(y)]

    data = LabelledCollection(X, y, classes_=sorted(np.unique(y)))

    # p=CounterExample(1, 8, 10, 5)

    # p = APP(data, sample_size=10, n_prevalences=11, random_seed=42)
    # p = NPP(data, sample_size=10, repeats=10, random_seed=42)
    # p = NPP(data, sample_size=10, repeats=10)
    p = USimplexPP(data, sample_size=10, repeats=10)

    for _ in range(2):
        print('init generator', p.__class__.__name__)
        for i in p():
            # print(i)
            print(i.instances, i.labels, i.prevalence())

    print('done')

