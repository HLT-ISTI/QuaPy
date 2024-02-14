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
    """
    Abstract parent class for sample generation protocols.
    """

    @abstractmethod
    def __call__(self):
        """
        Implements the protocol. Yields one sample at a time along with its prevalence

        :return: yields a tuple `(sample, prev) at a time, where `sample` is a set of instances
            and in which `prev` is an `nd.array` with the class prevalence values
        """
        ...

    def total(self):
        """
        Indicates the total number of samples that the protocol generates.

        :return: The number of samples to generate if known, or `None` otherwise.
        """
        return None


class IterateProtocol(AbstractProtocol):
    """
    A very simple protocol which simply iterates over a list of previously generated samples

    :param samples: a list of :class:`quapy.data.base.LabelledCollection`
    """
    def __init__(self, samples: [LabelledCollection]):
        self.samples = samples

    def __call__(self):
        """
        Yields one sample from the initial list at a time

        :return: yields a tuple `(sample, prev) at a time, where `sample` is a set of instances
            and in which `prev` is an `nd.array` with the class prevalence values
        """
        for sample in self.samples:
            yield sample.Xp

    def total(self):
        """
        Returns the number of samples in this protocol

        :return: int
        """
        return len(self.samples)


class AbstractStochasticSeededProtocol(AbstractProtocol):
    """
    An `AbstractStochasticSeededProtocol` is a protocol that generates, via any random procedure (e.g.,
    via random sampling), sequences of :class:`quapy.data.base.LabelledCollection` samples.
    The protocol abstraction enforces
    the object to be instantiated using a seed, so that the sequence can be fully replicated.
    In order to make this functionality possible, the classes extending this abstraction need to
    implement only two functions, :meth:`samples_parameters` which generates all the parameters
    needed for extracting the samples, and :meth:`sample` that, given some parameters as input,
    deterministically generates a sample.

    :param random_state: the seed for allowing to replicate any sequence of samples. Default is 0, meaning that
        the sequence will be consistent every time the protocol is called.
    """

    _random_state = -1  # means "not set"

    def __init__(self, random_state=0):
        self.random_state = random_state

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        self._random_state = random_state

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
        """
        Yields one sample at a time. The type of object returned depends on the `collator` function. The
        default behaviour returns tuples of the form `(sample, prevalence)`.

        :return: a tuple `(sample, prevalence)` if  return_type='sample_prev', or an instance of
            :class:`qp.data.LabelledCollection` if return_type='labelled_collection'
        """
        with ExitStack() as stack:
            if self.random_state == -1:
                raise ValueError('The random seed has never been initialized. '
                                 'Set it to None not to impose replicability.')
            if self.random_state is not None:
                stack.enter_context(qp.util.temp_seed(self.random_state))
            for params in self.samples_parameters():
                yield self.collator(self.sample(params))

    def collator(self, sample, *args):
        """
        The collator prepares the sample to accommodate the desired output format before returning the output.
        This collator simply returns the sample as it is. Classes inheriting from this abstract class can
        implement their custom collators.

        :param sample: the sample to be returned
        :param args: additional arguments
        :return: the sample adhering to a desired output format (in this case, the sample is returned as it is)
        """
        return sample


class OnLabelledCollectionProtocol:
    """
    Protocols that generate samples from a :class:`qp.data.LabelledCollection` object.
    """

    RETURN_TYPES = ['sample_prev', 'labelled_collection', 'index']

    def get_labelled_collection(self):
        """
        Returns the labelled collection on which this protocol acts.

        :return: an object of type :class:`qp.data.LabelledCollection`
        """
        return self.data

    def on_preclassified_instances(self, pre_classifications, in_place=False):
        """
        Returns a copy of this protocol that acts on a modified version of the original
        :class:`qp.data.LabelledCollection` in which the original instances have been replaced
        with the outputs of a classifier for each instance. (This is convenient for speeding-up
        the evaluation procedures for many samples, by pre-classifying the instances in advance.)

        :param pre_classifications: the predictions issued by a classifier, typically an array-like
            with shape `(n_instances,)` when the classifier is a hard one, or with shape
            `(n_instances, n_classes)` when the classifier is a probabilistic one.
        :param in_place: whether or not to apply the modification in-place or in a new copy (default).
        :return: a copy of this protocol
        """
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
        """
        Returns a collator function, i.e., a function that prepares the yielded data

        :param return_type: either 'sample_prev' (default) if the collator is requested to yield tuples of
            `(sample, prevalence)`, or 'labelled_collection' when it is requested to yield instances of
            :class:`qp.data.LabelledCollection`
        :return: the collator function (a callable function that takes as input an instance of
            :class:`qp.data.LabelledCollection`)
        """
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
    :param sample_size: integer, number of instances in each sample; if None (default) then it is taken from
        qp.environ["SAMPLE_SIZE"]. If this is not set, a ValueError exception is raised.
    :param n_prevalences: the number of equidistant prevalence points to extract from the [0,1] interval for the
        grid (default is 21)
    :param repeats: number of copies for each valid prevalence vector (default is 10)
    :param smooth_limits_epsilon: the quantity to add and subtract to the limits 0 and 1
    :param random_state: allows replicating samples across runs (default 0, meaning that the sequence of samples
        will be the same every time the protocol is called)
    :param sanity_check: int, raises an exception warning the user that the number of examples to be generated exceed
        this number; set to None for skipping this check
    :param return_type: set to "sample_prev" (default) to get the pairs of (sample, prevalence) at each iteration, or
        to "labelled_collection" to get instead instances of LabelledCollection
    """

    def __init__(self, data: LabelledCollection, sample_size=None, n_prevalences=21, repeats=10,
                 smooth_limits_epsilon=0, random_state=0, sanity_check=10000, return_type='sample_prev'):
        super(APP, self).__init__(random_state)
        self.data = data
        self.sample_size = qp._get_sample_size(sample_size)
        self.n_prevalences = n_prevalences
        self.repeats = repeats
        self.smooth_limits_epsilon = smooth_limits_epsilon
        if not ((isinstance(sanity_check, int) and sanity_check>0) or sanity_check is None):
            raise ValueError('param "sanity_check" must either be None or a positive integer')
        if isinstance(sanity_check, int):
            n = F.num_prevalence_combinations(n_prevpoints=n_prevalences, n_classes=data.n_classes, n_repeats=repeats)
            if n > sanity_check:
                raise RuntimeError(
                    f"Abort: the number of samples that will be generated by {self.__class__.__name__} ({n}) "
                    f"exceeds the maximum number of allowed samples ({sanity_check = }). Set 'sanity_check' to "
                    f"None, or to a higher number, for bypassing this check.")

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
        s = F.prevalence_linspace(self.n_prevalences, repeats=1, smooth_limits_epsilon=self.smooth_limits_epsilon)
        eps = (s[1]-s[0])/2 # handling floating rounding
        s = [s] * (dimensions - 1)
        prevs = [p for p in itertools.product(*s, repeat=1) if (sum(p) < (1.+eps))]
        prevs = np.asarray(prevs).reshape(len(prevs), -1)
        if self.repeats > 1:
            prevs = np.repeat(prevs, self.repeats, axis=0)
        return prevs

    def samples_parameters(self):
        """
        Return all the necessary parameters to replicate the samples as according to the APP protocol.

        :return: a list of indexes that realize the APP sampling
        """
        indexes = []
        for prevs in self.prevalence_grid():
            index = self.data.sampling_index(self.sample_size, *prevs)
            indexes.append(index)
        return indexes

    def sample(self, index):
        """
        Realizes the sample given the index of the instances.

        :param index: indexes of the instances to select
        :return: an instance of :class:`qp.data.LabelledCollection`
        """
        return self.data.sampling_from_index(index)

    def total(self):
        """
        Returns the number of samples that will be generated

        :return: int
        """
        return F.num_prevalence_combinations(self.n_prevalences, self.data.n_classes, self.repeats)


class NPP(AbstractStochasticSeededProtocol, OnLabelledCollectionProtocol):
    """
    A generator of samples that implements the natural prevalence protocol (NPP). The NPP consists of drawing
    samples uniformly at random, therefore approximately preserving the natural prevalence of the collection.

    :param data: a `LabelledCollection` from which the samples will be drawn
    :param sample_size: integer, the number of instances in each sample; if None (default) then it is taken from
        qp.environ["SAMPLE_SIZE"]. If this is not set, a ValueError exception is raised.
    :param repeats: the number of samples to generate. Default is 100.
    :param random_state: allows replicating samples across runs (default 0, meaning that the sequence of samples
        will be the same every time the protocol is called)
    :param return_type: set to "sample_prev" (default) to get the pairs of (sample, prevalence) at each iteration, or
        to "labelled_collection" to get instead instances of LabelledCollection
    """

    def __init__(self, data:LabelledCollection, sample_size=None, repeats=100, random_state=0,
                 return_type='sample_prev'):
        super(NPP, self).__init__(random_state)
        self.data = data
        self.sample_size = qp._get_sample_size(sample_size)
        self.repeats = repeats
        self.random_state = random_state
        self.collator = OnLabelledCollectionProtocol.get_collator(return_type)

    def samples_parameters(self):
        """
        Return all the necessary parameters to replicate the samples as according to the NPP protocol.

        :return: a list of indexes that realize the NPP sampling
        """
        indexes = []
        for _ in range(self.repeats):
            index = self.data.uniform_sampling_index(self.sample_size)
            indexes.append(index)
        return indexes

    def sample(self, index):
        """
        Realizes the sample given the index of the instances.

        :param index: indexes of the instances to select
        :return: an instance of :class:`qp.data.LabelledCollection`
        """
        return self.data.sampling_from_index(index)

    def total(self):
        """
        Returns the number of samples that will be generated (equals to "repeats")

        :return: int
        """
        return self.repeats


class UPP(AbstractStochasticSeededProtocol, OnLabelledCollectionProtocol):
    """
    A variant of :class:`APP` that, instead of using a grid of equidistant prevalence values,
    relies on the Kraemer algorithm for sampling unit (k-1)-simplex uniformly at random, with
    k the number of classes. This protocol covers the entire range of prevalence values in a
    statistical sense, i.e., unlike APP there is no guarantee that it is covered precisely
    equally for all classes, but it is preferred in cases in which the number of possible
    combinations of the grid values of APP makes this endeavour intractable.

    :param data: a `LabelledCollection` from which the samples will be drawn
    :param sample_size: integer, the number of instances in each sample; if None (default) then it is taken from
        qp.environ["SAMPLE_SIZE"]. If this is not set, a ValueError exception is raised.
    :param repeats: the number of samples to generate. Default is 100.
    :param random_state: allows replicating samples across runs (default 0, meaning that the sequence of samples
        will be the same every time the protocol is called)
    :param return_type: set to "sample_prev" (default) to get the pairs of (sample, prevalence) at each iteration, or
        to "labelled_collection" to get instead instances of LabelledCollection
    """

    def __init__(self, data: LabelledCollection, sample_size=None, repeats=100, random_state=0,
                 return_type='sample_prev'):
        super(UPP, self).__init__(random_state)
        self.data = data
        self.sample_size = qp._get_sample_size(sample_size)
        self.repeats = repeats
        self.random_state = random_state
        self.collator = OnLabelledCollectionProtocol.get_collator(return_type)

    def samples_parameters(self):
        """
        Return all the necessary parameters to replicate the samples as according to the UPP protocol.

        :return: a list of indexes that realize the UPP sampling
        """
        indexes = []
        for prevs in F.uniform_simplex_sampling(n_classes=self.data.n_classes, size=self.repeats):
            index = self.data.sampling_index(self.sample_size, *prevs)
            indexes.append(index)
        return indexes

    def sample(self, index):
        """
        Realizes the sample given the index of the instances.

        :param index: indexes of the instances to select
        :return: an instance of :class:`qp.data.LabelledCollection`
        """
        return self.data.sampling_from_index(index)

    def total(self):
        """
        Returns the number of samples that will be generated (equals to "repeats")

        :return: int
        """
        return self.repeats


class DomainMixer(AbstractStochasticSeededProtocol):
    """
    Generates mixtures of two domains (A and B) at controlled rates, but preserving the original class prevalence.

    :param domainA: one domain, an object of :class:`qp.data.LabelledCollection`
    :param domainB: another domain, an object of :class:`qp.data.LabelledCollection`
    :param sample_size: integer, the number of instances in each sample; if None (default) then it is taken from
        qp.environ["SAMPLE_SIZE"]. If this is not set, a ValueError exception is raised.
    :param repeats: int, number of samples to draw for every mixture rate
    :param prevalence: the prevalence to preserv along the mixtures. If specified, should be an array containing
        one prevalence value (positive float) for each class and summing up to one. If not specified, the prevalence
        will be taken from the domain A (default).
    :param mixture_points: an integer indicating the number of points to take from a linear scale (e.g., 21 will
        generate the mixture points [1, 0.95, 0.9, ..., 0]), or the array of mixture values itself.
        the specific points
    :param random_state: allows replicating samples across runs (default 0, meaning that the sequence of samples
        will be the same every time the protocol is called)
    """

    def __init__(
            self,
            domainA: LabelledCollection,
            domainB: LabelledCollection,
            sample_size,
            repeats=1,
            prevalence=None,
            mixture_points=11,
            random_state=0,
            return_type='sample_prev'):
        super(DomainMixer, self).__init__(random_state)
        self.A = domainA
        self.B = domainB
        self.sample_size = qp._get_sample_size(sample_size)
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
        self.random_state = random_state
        self.collator = OnLabelledCollectionProtocol.get_collator(return_type)

    def samples_parameters(self):
        """
        Return all the necessary parameters to replicate the samples as according to the this protocol.

        :return: a list of zipped indexes (from A and B) that realize the sampling
        """
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
        """
        Realizes the sample given a pair of indexes of the instances from A and B.

        :param indexes: indexes of the instances to select from A and B
        :return: an instance of :class:`qp.data.LabelledCollection`
        """
        indexesA, indexesB = indexes
        sampleA = self.A.sampling_from_index(indexesA)
        sampleB = self.B.sampling_from_index(indexesB)
        return sampleA+sampleB

    def total(self):
        """
        Returns the number of samples that will be generated (equals to "repeats * mixture_points")

        :return: int
        """
        return self.repeats * len(self.mixture_points)


# aliases

ArtificialPrevalenceProtocol = APP
NaturalPrevalenceProtocol = NPP
UniformPrevalenceProtocol = UPP