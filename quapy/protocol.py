import numpy as np
import itertools
from collections.abc import Generator
from contextlib import ExitStack
from abc import ABCMeta, abstractmethod

from quapy.data import LabelledCollection
import quapy.functional as F
from tqdm import tqdm


# 0.1.7
# change the LabelledCollection API (removing protocol-related samplings)
# need to change the two references to the above in the wiki / doc, and code examples...
# removed artificial_prevalence_sampling from functional

# maybe add some parameters in the init of the protocols (or maybe only for IndexableWhateverProtocols
# indicating that the protocol should return indexes, and not samples themselves?
# also: some parameters in the init could be used to indicate that the method should return a tuple with
# unlabelled instances and the vector of prevalence values (and not a LabelledCollection).
# Or: this can be done in a different function; i.e., we use one function (now __call__) to return
# LabelledCollections, and another new one for returning the other output, which is more general for
# evaluation purposes.

# the so-called "gen" function has to be implemented as a protocol. The problem here is that this function
# should be able to return only unlabelled instances plus a vector of prevalences (and not LabelledCollections).
# This was coded as different functions in 0.1.6


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


class APP(AbstractStochasticSeededProtocol):
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

    def __init__(self, data:LabelledCollection, sample_size, n_prevalences=21, repeats=10, random_seed=None):
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

    def total(self):
        return F.num_prevalence_combinations(self.n_prevalences, self.data.n_classes, self.repeats)


class NPP(AbstractStochasticSeededProtocol):
    """
    A generator of samples that implements the natural prevalence protocol (NPP). The NPP consists of drawing
    samples uniformly at random, therefore approximately preserving the natural prevalence of the collection.

    :param data: a `LabelledCollection` from which the samples will be drawn
    :param sample_size: integer, the number of instances in each sample
    :param repeats: the number of samples to generate. Default is 100.
    :param random_seed: allows replicating samples across runs (default None)
    """

    def __init__(self, data:LabelledCollection, sample_size, repeats=100, random_seed=None):
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

    def total(self):
        return self.repeats



class USimplexPP(AbstractStochasticSeededProtocol):
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

    def __init__(self, data: LabelledCollection, sample_size, repeats=100, random_seed=None):
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

    def total(self):
        return self.repeats



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
            random_seed=None):
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




if __name__=='__main__':
    import numpy as np
    import quapy as qp

    # domainA
    y = [0]*25 + [1]*25 + [2]*25 + [3]*25
    X = ['A:'+str(i)+'-'+str(yi) for i, yi in enumerate(y)]
    data = LabelledCollection(X, y, classes_=sorted(np.unique(y)))

    # domain B
    y = [0]*25 + [1]*25 + [2]*25 + [3]*25
    X = ['B:'+str(i)+'-'+str(yi) for i, yi in enumerate(y)]
    dataB = LabelledCollection(X, y, classes_=sorted(np.unique(y)))

    # p = APP(data, sample_size=10, n_prevalences=11, random_seed=42)
    # p = NPP(data, sample_size=10, repeats=10, random_seed=42)
    # p = NPP(data, sample_size=10, repeats=10)
    # p = USimplexPP(data, sample_size=10, repeats=10)
    p = CovariateShiftPP(data, dataB, sample_size=10, mixture_points=11, random_seed=1)

    for _ in range(2):
        print('init generator', p.__class__.__name__)
        for i in tqdm(p(), total=p.total()):
            # print(i)
            print(i.instances, i.labels, i.prevalence())

    print('done')

