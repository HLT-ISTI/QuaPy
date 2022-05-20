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


class NewAbstractProtocol(metaclass=Generator):
    @abstractmethod
    def send(self, value):
        """Send a value into the generator.
        Return next yielded value or raise StopIteration.
        """
        raise StopIteration

    @abstractmethod
    def throw(self, typ, val=None, tb=None):
        """Raise an exception in the generator.
        Return next yielded value or raise StopIteration.
        """
        if val is None:
            if tb is None:
                raise typ
            val = typ()
        if tb is not None:
            val = val.with_traceback(tb)
        raise val



class AbstractProtocol(metaclass=ABCMeta):
    """
    Abstract class for sampling protocols.
    A sampling protocol defines how to generate samples out of some dataset.
    """

    def __call__(self):
        """
        A generator that yields one sample at each iteration

        :return: yield one sample (instance of :class:`quapy.data.LabelledCollection`) at each iteration
        """
        for index in self.indexes(data):
            yield data.sampling_from_index(index)

    def indexes(self, data: LabelledCollection):
        """
        A generator that yields one sample index at each iteration.
        (This function is mainly a generic decorator that sets, if requested, the local random seed; the real
        sampling is implemented by :meth:`_indexes`.)

        :param data: the set of data from which samples' indexes are to be drawn
        :return: one sample index (instance of `np.ndarray`) at each iteration
        """
        with ExitStack() as stack:
            if self.random_seed is not None:
                stack.enter_context(qp.util.temp_seed(self.random_seed))
            for index in self._indexes(data):
                yield index

    @abstractmethod
    def _indexes(self, data: LabelledCollection):
        ...


class APP(AbstractProtocol):
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
        self.data = data
        self.sample_size = sample_size
        self.n_prevalences = n_prevalences
        self.repeats = repeats
        self.random_seed = random_seed

    def _indexes(self, data: LabelledCollection):
        for prevs in self.prevalence_grid(dimensions=data.n_classes):
            yield data.sampling_index(self.sample_size, *prevs)

    def prevalence_grid(self, dimensions, return_constrained_dim=False):
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
        :param return_constrained_dim: set to True to return all dimensions, or to False (default) for ommitting the
            constrained dimension
        :return: a `np.ndarray` of shape `(n, dimensions)` if `return_constrained_dim=True` or of shape
            `(n, dimensions-1)` if `return_constrained_dim=False`, where `n` is the number of valid combinations found
            in the grid multiplied by `repeat`
        """
        s = np.linspace(0., 1., self.n_prevalences, endpoint=True)
        s = [s] * (dimensions - 1)
        prevs = [p for p in itertools.product(*s, repeat=1) if sum(p) <= 1]
        if return_constrained_dim:
            prevs = [p + (1 - sum(p),) for p in prevs]
        prevs = np.asarray(prevs).reshape(len(prevs), -1)
        if self.repeats > 1:
            prevs = np.repeat(prevs, self.repeats, axis=0)
        return prevs


class NPP(AbstractProtocol):
    """
    A generator of samples that implements the natural prevalence protocol (NPP). The NPP consists of drawing
    samples uniformly at random, therefore approximately preserving the natural prevalence of the collection.

    :param sample_size: integer, the number of instances in each sample
    :param repeats: the number of samples to generate
    """

    def __init__(self, sample_size, repeats=1, random_seed=None):
        self.sample_size = sample_size
        self.repeats = repeats
        self.random_seed = random_seed

    def _indexes(self, data: LabelledCollection):
        for _ in range(self.repeats):
            yield data.uniform_sampling_index(self.sample_size)


class USimplexPP(AbstractProtocol):

    def __init__(self, sample_size, repeats=1, random_seed=None):
        self.sample_size = sample_size
        self.repeats = repeats
        self.random_seed = random_seed

    def _indexes(self, data: LabelledCollection):
        for prevs in F.uniform_simplex_sampling(n_classes=data.n_classes, size=self.repeats):
            yield data.sampling_index(self.sample_size, *prevs)



if __name__=='__main__':
    import numpy as np
    import quapy as qp

    y = [0]*25 + [1]*25 + [2]*25 + [3]*25
    X = [str(i)+'-'+str(yi) for i, yi in enumerate(y)]

    data = LabelledCollection(X, y, classes_=sorted(np.unique(y)))

    # p = APP(10, n_prevalences=11, random_seed=42)
    # p = NPP(10, repeats=10, random_seed=42)
    p = USimplexPP(10, repeats=10, random_seed=42)

    for i in p(data):
        print(i.instances, i.classes, i.prevalence())

    print('done')

