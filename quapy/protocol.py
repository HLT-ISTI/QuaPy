from copy import deepcopy
import quapy as qp
import numpy as np
import itertools
from contextlib import ExitStack
from abc import ABCMeta, abstractmethod
from quapy.data import LabelledCollection
import quapy.functional as F
from os.path import exists
from glob import glob


class AbstractProtocol(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self):
        """
        Implements the protocol. Yields one sample at a time

        :return: yields one sample at a time
        """
        ...

    def total(self):
        """
        Indicates the total number of samples that the protocol generates.

        :return: The number of samples to generate if known, or `None` otherwise.
        """
        return None


class AbstractStochasticSeededProtocol(AbstractProtocol):
    """
    An AbstractStochasticSeededProtocol is a protocol that generates, via any random procedure (e.g.,
    via random sapling), sequences of `LabelledCollection` samples. The protocol abstraction enforces
    the object to be instantiated using a seed, so that the sequence can be completely replicated.
    In order to make this functionality possible, the classes extending this abstraction need to
    implement only two functions, :meth:`samples_parameters` which generates all the parameters
    needed for extracting the samples, and :meth:`sample` that, given some parameters as input,
    deterministically generates a sample.

    :param seed: the seed for allowing to replicate any sequence of samples. Default is None, meaning that
        the sequence will be different every time the protocol is called.
    """

    _random_seed = -1  # means "not set"

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
            if self.random_seed == -1:
                raise ValueError('The random seed has never been initialized. '
                                 'Set it to None not to impose replicability.')
            if self.random_seed is not None:
                stack.enter_context(qp.util.temp_seed(self.random_seed))
            for params in self.samples_parameters():
                yield self.collator(self.sample(params))

    def collator(self, sample, *args):
        return sample


class OnLabelledCollectionProtocol:

    RETURN_TYPES = ['sample_prev', 'labelled_collection']

    def get_labelled_collection(self):
        return self.data

    def on_preclassified_instances(self, pre_classifications, in_place=False):
        assert len(pre_classifications) == len(self.data), \
            f'error: the pre-classified data has different shape ' \
            f'(expected {len(self.data)}, found {len(pre_classifications)})'
        if in_place:
            self.data.instances = pre_classifications
            return self
        else:
            new = deepcopy(self)
            return new.on_preclassified_instances(pre_classifications, in_place=True)

    @classmethod
    def get_collator(cls, return_type='sample_prev'):
        assert return_type in cls.RETURN_TYPES, \
            f'unknown return type passed as argument; valid ones are {cls.RETURN_TYPES}'
        if return_type=='sample_prev':
            return lambda lc:lc.Xp
        elif return_type=='labelled_collection':
            return lambda lc:lc


class APP(AbstractStochasticSeededProtocol, OnLabelledCollectionProtocol):
    """
    Implementation of the artificial prevalence protocol (APP).
    The APP consists of exploring a grid of prevalence values containing `n_prevalences` points (e.g.,
    [0, 0.05, 0.1, 0.15, ..., 1], if `n_prevalences=21`), and generating all valid combinations of
    prevalence values for all classes (e.g., for 3 classes, samples with [0, 0, 1], [0, 0.05, 0.95], ...,
    [1, 0, 0] prevalence values of size `sample_size` will be yielded). The number of samples for each valid
    combination of prevalence values is indicated by `repeats`.

    :param data: a `LabelledCollection` from which the samples will be drawn
    :param sample_size: integer, number of instances in each sample
    :param n_prevalences: the number of equidistant prevalence points to extract from the [0,1] interval for the
        grid (default is 21)
    :param repeats: number of copies for each valid prevalence vector (default is 10)
    :param random_seed: allows replicating samples across runs (default None)
    """

    def __init__(self, data:LabelledCollection, sample_size, n_prevalences=21, repeats=10, random_seed=None, return_type='sample_prev'):
        super(APP, self).__init__(random_seed)
        self.data = data
        self.sample_size = sample_size
        self.n_prevalences = n_prevalences
        self.repeats = repeats
        self.collator = OnLabelledCollectionProtocol.get_collator(return_type)

    def prevalence_grid(self):
        """
        Generates vectors of prevalence values from an exhaustive grid of prevalence values. The
        number of prevalence values explored for each dimension depends on `n_prevalences`, so that, if, for example,
        `n_prevalences=11` then the prevalence values of the grid are taken from [0, 0.1, 0.2, ..., 0.9, 1]. Only
        valid prevalence distributions are returned, i.e., vectors of prevalence values that sum up to 1. For each
        valid vector of prevalence values, `repeat` copies are returned. The vector of prevalence values can be
        implicit (by setting `return_constrained_dim=False`), meaning that the last dimension (which is constrained
        to 1 - sum of the rest) is not returned (note that, quite obviously, in this case the vector does not sum up to
        1). Note that this method is deterministic, i.e., there is no random sampling anywhere.

        :return: a `np.ndarray` of shape `(n, dimensions)` if `return_constrained_dim=True` or of shape
            `(n, dimensions-1)` if `return_constrained_dim=False`, where `n` is the number of valid combinations found
            in the grid multiplied by `repeat`
        """
        dimensions = self.data.n_classes
        s = np.linspace(0., 1., self.n_prevalences, endpoint=True)
        s = [s] * (dimensions - 1)
        prevs = [p for p in itertools.product(*s, repeat=1) if (sum(p) <= 1.0)]
        prevs = np.asarray(prevs).reshape(len(prevs), -1)
        if self.repeats > 1:
            prevs = np.repeat(prevs, self.repeats, axis=0)
        return prevs

    def samples_parameters(self):
        indexes = []
        for prevs in self.prevalence_grid():
            index = self.data.sampling_index(self.sample_size, *prevs)
            indexes.append(index)
        return indexes

    def sample(self, index):
        return self.data.sampling_from_index(index)

    def total(self):
        return F.num_prevalence_combinations(self.n_prevalences, self.data.n_classes, self.repeats)


class NPP(AbstractStochasticSeededProtocol, OnLabelledCollectionProtocol):
    """
    A generator of samples that implements the natural prevalence protocol (NPP). The NPP consists of drawing
    samples uniformly at random, therefore approximately preserving the natural prevalence of the collection.

    :param data: a `LabelledCollection` from which the samples will be drawn
    :param sample_size: integer, the number of instances in each sample
    :param repeats: the number of samples to generate. Default is 100.
    :param random_seed: allows replicating samples across runs (default None)
    """

    def __init__(self, data:LabelledCollection, sample_size, repeats=100, random_seed=None, return_type='sample_prev'):
        super(NPP, self).__init__(random_seed)
        self.data = data
        self.sample_size = sample_size
        self.repeats = repeats
        self.random_seed = random_seed
        self.collator = OnLabelledCollectionProtocol.get_collator(return_type)

    def samples_parameters(self):
        indexes = []
        for _ in range(self.repeats):
            index = self.data.uniform_sampling_index(self.sample_size)
            indexes.append(index)
        return indexes

    def sample(self, index):
        return self.data.sampling_from_index(index)

    def total(self):
        return self.repeats


class USimplexPP(AbstractStochasticSeededProtocol, OnLabelledCollectionProtocol):
    """
    A variant of :class:`APP` that, instead of using a grid of equidistant prevalence values,
    relies on the Kraemer algorithm for sampling unit (k-1)-simplex uniformly at random, with
    k the number of classes. This protocol covers the entire range of prevalence values in a
    statistical sense, i.e., unlike APP there is no guarantee that it is covered precisely
    equally for all classes, but it is preferred in cases in which the number of possible
    combinations of the grid values of APP makes this endeavour intractable.

    :param data: a `LabelledCollection` from which the samples will be drawn
    :param sample_size: integer, the number of instances in each sample
    :param repeats: the number of samples to generate. Default is 100.
    :param random_seed: allows replicating samples across runs (default None)
    """

    def __init__(self, data: LabelledCollection, sample_size, repeats=100, random_seed=None, return_type='sample_prev'):
        super(USimplexPP, self).__init__(random_seed)
        self.data = data
        self.sample_size = sample_size
        self.repeats = repeats
        self.random_seed = random_seed
        self.collator = OnLabelledCollectionProtocol.get_collator(return_type)

    def samples_parameters(self):
        indexes = []
        for prevs in F.uniform_simplex_sampling(n_classes=self.data.n_classes, size=self.repeats):
            index = self.data.sampling_index(self.sample_size, *prevs)
            indexes.append(index)
        return indexes

    def sample(self, index):
        return self.data.sampling_from_index(index)

    def total(self):
        return self.repeats


# class LoadSamplesFromDirectory(AbstractProtocol):
#
#     def __init__(self, folder_path, loader_fn, classes=None, **loader_kwargs):
#         assert exists(folder_path), f'folder {folder_path} does not exist'
#         assert callable(loader_fn), f'the passed load_fn does not seem to be callable'
#         self.folder_path = folder_path
#         self.loader_fn = loader_fn
#         self.classes = classes
#         self.loader_kwargs = loader_kwargs
#         self._list_files = None
#
#     def __call__(self):
#         for file in self.list_files:
#             yield LabelledCollection.load(file, loader_func=self.loader_fn, classes=self.classes, **self.loader_kwargs)
#
#     @property
#     def list_files(self):
#         if self._list_files is None:
#             self._list_files = sorted(glob(self.folder_path, '*'))
#         return self._list_files
#
#     def total(self):
#         return len(self.list_files)


class CovariateShiftPP(AbstractStochasticSeededProtocol):
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
            random_seed=None,
            return_type='sample_prev'):
        super(CovariateShiftPP, self).__init__(random_seed)
        self.A = domainA
        self.B = domainB
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
        if isinstance(mixture_points, int):
            self.mixture_points = np.linspace(0, 1, mixture_points)[::-1]
        else:
            self.mixture_points = np.asarray(mixture_points)
            assert all(np.logical_and(self.mixture_points >= 0, self.mixture_points<=1)), \
                'mixture_model datatype not understood (expected int or a sequence of real values in [0,1])'
        self.random_seed = random_seed
        self.collator = OnLabelledCollectionProtocol.get_collator(return_type)

    def samples_parameters(self):
        indexesA, indexesB = [], []
        for propA in self.mixture_points:
            for _ in range(self.repeats):
                nA = int(np.round(self.sample_size * propA))
                nB = self.sample_size-nA
                sampleAidx = self.A.sampling_index(nA, *self.prevalence)
                sampleBidx = self.B.sampling_index(nB, *self.prevalence)
                indexesA.append(sampleAidx)
                indexesB.append(sampleBidx)
        return list(zip(indexesA, indexesB))

    def sample(self, indexes):
        indexesA, indexesB = indexes
        sampleA = self.A.sampling_from_index(indexesA)
        sampleB = self.B.sampling_from_index(indexesB)
        return sampleA+sampleB

    def total(self):
        return self.repeats * len(self.mixture_points)


