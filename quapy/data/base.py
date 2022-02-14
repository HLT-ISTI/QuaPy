import numpy as np
from scipy.sparse import issparse
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

from quapy.functional import artificial_prevalence_sampling, strprev


class LabelledCollection:
    """
    A LabelledCollection is a set of objects each with a label associated to it. This class implements many sampling
    routines.

    :param instances: array-like (np.ndarray, list, or csr_matrix are supported)
    :param labels: array-like with the same length of instances
    :param classes_: optional, list of classes from which labels are taken. If not specified, the classes are inferred
        from the labels. The classes must be indicated in cases in which some of the labels might have no examples
        (i.e., a prevalence of 0)
    """

    def __init__(self, instances, labels, classes_=None):
        if issparse(instances):
            self.instances = instances
        elif isinstance(instances, list) and len(instances) > 0 and isinstance(instances[0], str):
            # lists of strings occupy too much as ndarrays (although python-objects add a heavy overload)
            self.instances = np.asarray(instances, dtype=object)
        else:
            self.instances = np.asarray(instances)
        self.labels = np.asarray(labels)
        n_docs = len(self)
        if classes_ is None:
            self.classes_ = np.unique(self.labels)
            self.classes_.sort()
        else:
            self.classes_ = np.unique(np.asarray(classes_))
            self.classes_.sort()
            if len(set(self.labels).difference(set(classes_))) > 0:
                raise ValueError(f'labels ({set(self.labels)}) contain values not included in classes_ ({set(classes_)})')
        self.index = {class_: np.arange(n_docs)[self.labels == class_] for class_ in self.classes_}

    @classmethod
    def load(cls, path: str, loader_func: callable, classes=None, **loader_kwargs):
        """
        Loads a labelled set of data and convert it into a :class:`LabelledCollection` instance. The function in charge
        of reading the instances must be specified. This function can be a custom one, or any of the reading functions
        defined in :mod:`quapy.data.reader` module.

        :param path: string, the path to the file containing the labelled instances
        :param loader_func: a custom function that implements the data loader and returns a tuple with instances and
            labels
        :param classes: array-like, the classes according to which the instances are labelled
        :param loader_kwargs: any argument that the `loader_func` function needs in order to read the instances, i.e.,
            these arguments are used to call `loader_func(path, **loader_kwargs)`
        :return: a :class:`LabelledCollection` object
        """
        return LabelledCollection(*loader_func(path, **loader_kwargs), classes)

    def __len__(self):
        """
        Returns the length of this collection (number of labelled instances)

        :return: integer
        """
        return self.instances.shape[0]

    def prevalence(self):
        """
        Returns the prevalence, or relative frequency, of the classes of interest.

        :return: a np.ndarray of shape `(n_classes)` with the relative frequencies of each class, in the same order
            as listed by `self.classes_`
        """
        return self.counts() / len(self)

    def counts(self):
        """
        Returns the number of instances for each of the classes of interest.

        :return: a np.ndarray of shape `(n_classes)` with the number of instances of each class, in the same order
            as listed by `self.classes_`
        """
        return np.asarray([len(self.index[class_]) for class_ in self.classes_])

    @property
    def n_classes(self):
        """
        The number of classes

        :return: integer
        """
        return len(self.classes_)

    @property
    def binary(self):
        """
        Returns True if the number of classes is 2

        :return: boolean
        """
        return self.n_classes == 2

    def sampling_index(self, size, *prevs, shuffle=True):
        """
        Returns an index to be used to extract a random sample of desired size and desired prevalence values. If the
        prevalence values are not specified, then returns the index of a uniform sampling.
        For each class, the sampling is drawn with replacement if the requested prevalence is larger than
        the actual prevalence of the class, or without replacement otherwise.

        :param size: integer, the requested size
        :param prevs: the prevalence for each class; the prevalence value for the last class can be lead empty since
            it is constrained. E.g., for binary collections, only the prevalence `p` for the first class (as listed in
            `self.classes_` can be specified, while the other class takes prevalence value `1-p`
        :param shuffle: if set to True (default), shuffles the index before returning it
        :return: a np.ndarray of shape `(size)` with the indexes
        """
        if len(prevs) == 0:  # no prevalence was indicated; returns an index for uniform sampling
            return self.uniform_sampling_index(size)
        if len(prevs) == self.n_classes - 1:
            prevs = prevs + (1 - sum(prevs),)
        assert len(prevs) == self.n_classes, 'unexpected number of prevalences'
        assert sum(prevs) == 1, f'prevalences ({prevs}) wrong range (sum={sum(prevs)})'

        taken = 0
        indexes_sample = []
        for i, class_ in enumerate(self.classes_):
            if i == self.n_classes - 1:
                n_requested = size - taken
            else:
                n_requested = int(size * prevs[i])

            n_candidates = len(self.index[class_])
            index_sample = self.index[class_][
                np.random.choice(n_candidates, size=n_requested, replace=(n_requested > n_candidates))
            ] if n_requested > 0 else []

            indexes_sample.append(index_sample)
            taken += n_requested

        indexes_sample = np.concatenate(indexes_sample).astype(int)

        if shuffle:
            indexes_sample = np.random.permutation(indexes_sample)

        return indexes_sample

    def uniform_sampling_index(self, size):
        """
        Returns an index to be used to extract a uniform sample of desired size. The sampling is drawn
        with replacement if the requested size is greater than the number of instances, or without replacement
        otherwise.

        :param size: integer, the size of the uniform sample
        :return: a np.ndarray of shape `(size)` with the indexes
        """
        return np.random.choice(len(self), size, replace=False)

    def sampling(self, size, *prevs, shuffle=True):
        """
        Return a random sample (an instance of :class:`LabelledCollection`) of desired size and desired prevalence
        values. For each class, the sampling is drawn without replacement if the requested prevalence is larger than
        the actual prevalence of the class, or with replacement otherwise.

        :param size: integer, the requested size
        :param prevs: the prevalence for each class; the prevalence value for the last class can be lead empty since
            it is constrained. E.g., for binary collections, only the prevalence `p` for the first class (as listed in
            `self.classes_` can be specified, while the other class takes prevalence value `1-p`
        :param shuffle: if set to True (default), shuffles the index before returning it
        :return: an instance of :class:`LabelledCollection` with length == `size` and prevalence close to `prevs` (or
            prevalence == `prevs` if the exact prevalence values can be met as proportions of instances)
        """
        prev_index = self.sampling_index(size, *prevs, shuffle=shuffle)
        return self.sampling_from_index(prev_index)

    def uniform_sampling(self, size):
        """
        Returns a uniform sample (an instance of :class:`LabelledCollection`) of desired size. The sampling is drawn
        with replacement if the requested size is greater than the number of instances, or without replacement
        otherwise.

        :param size: integer, the requested size
        :return: an instance of :class:`LabelledCollection` with length == `size`
        """
        unif_index = self.uniform_sampling_index(size)
        return self.sampling_from_index(unif_index)

    def sampling_from_index(self, index):
        """
        Returns an instance of :class:`LabelledCollection` whose elements are sampled from this collection using the
        index.

        :param index: np.ndarray
        :return: an instance of :class:`LabelledCollection`
        """
        documents = self.instances[index]
        labels = self.labels[index]
        return LabelledCollection(documents, labels, classes_=self.classes_)

    def split_stratified(self, train_prop=0.6, random_state=None):
        """
        Returns two instances of :class:`LabelledCollection` split with stratification from this collection, at desired
        proportion.

        :param train_prop: the proportion of elements to include in the left-most returned collection (typically used
            as the training collection). The rest of elements are included in the right-most returned collection
            (typically used as a test collection).
        :param random_state: if specified, guarantees reproducibility of the split.
        :return: two instances of :class:`LabelledCollection`, the first one with `train_prop` elements, and the
            second one with `1-train_prop` elements
        """
        tr_docs, te_docs, tr_labels, te_labels = \
            train_test_split(self.instances, self.labels, train_size=train_prop, stratify=self.labels,
                             random_state=random_state)
        return LabelledCollection(tr_docs, tr_labels), LabelledCollection(te_docs, te_labels)

    def artificial_sampling_generator(self, sample_size, n_prevalences=101, repeats=1):
        """
        A generator of samples that implements the artificial prevalence protocol (APP).
        The APP consists of exploring a grid of prevalence values containing `n_prevalences` points (e.g.,
        [0, 0.05, 0.1, 0.15, ..., 1], if `n_prevalences=21`), and generating all valid combinations of
        prevalence values for all classes (e.g., for 3 classes, samples with [0, 0, 1], [0, 0.05, 0.95], ...,
        [1, 0, 0] prevalence values of size `sample_size` will be yielded). The number of samples for each valid
        combination of prevalence values is indicated by `repeats`.

        :param sample_size: the number of instances in each sample
        :param n_prevalences: the number of prevalence points to be taken from the [0,1] interval (including the
            limits {0,1}). E.g., if `n_prevalences=11`, then the prevalence points to take are [0, 0.1, 0.2, ..., 1]
        :param repeats: the number of samples to generate for each valid combination of prevalence values (default 1)
        :return: yield samples generated at artificially controlled prevalence values
        """
        dimensions = self.n_classes
        for prevs in artificial_prevalence_sampling(dimensions, n_prevalences, repeats):
            yield self.sampling(sample_size, *prevs)

    def artificial_sampling_index_generator(self, sample_size, n_prevalences=101, repeats=1):
        """
        A generator of sample indexes implementing the artificial prevalence protocol (APP).
        The APP consists of exploring
        a grid of prevalence values (e.g., [0, 0.05, 0.1, 0.15, ..., 1]), and generating all valid combinations of
        prevalence values for all classes (e.g., for 3 classes, samples with [0, 0, 1], [0, 0.05, 0.95], ...,
        [1, 0, 0] prevalence values of size `sample_size` will be yielded). The number of sample indexes for each valid
        combination of prevalence values is indicated by `repeats`

        :param sample_size: the number of instances in each sample (i.e., length of each index)
        :param n_prevalences: the number of prevalence points to be taken from the [0,1] interval (including the
            limits {0,1}). E.g., if `n_prevalences=11`, then the prevalence points to take are [0, 0.1, 0.2, ..., 1]
        :param repeats: the number of samples to generate for each valid combination of prevalence values (default 1)
        :return: yield the indexes that generate the samples according to APP
        """
        dimensions = self.n_classes
        for prevs in artificial_prevalence_sampling(dimensions, n_prevalences, repeats):
            yield self.sampling_index(sample_size, *prevs)

    def natural_sampling_generator(self, sample_size, repeats=100):
        """
        A generator of samples that implements the natural prevalence protocol (NPP). The NPP consists of drawing
        samples uniformly at random, therefore approximately preserving the natural prevalence of the collection.

        :param sample_size: integer, the number of instances in each sample
        :param repeats: the number of samples to generate
        :return: yield instances of :class:`LabelledCollection`
        """
        for _ in range(repeats):
            yield self.uniform_sampling(sample_size)

    def natural_sampling_index_generator(self, sample_size, repeats=100):
        """
        A generator of sample indexes according to the natural prevalence protocol (NPP). The NPP consists of drawing
        samples uniformly at random, therefore approximately preserving the natural prevalence of the collection.

        :param sample_size: integer, the number of instances in each sample (i.e., the length of each index)
        :param repeats: the number of indexes to generate
        :return: yield `repeats` instances of np.ndarray with shape `(sample_size,)`
        """
        for _ in range(repeats):
            yield self.uniform_sampling_index(sample_size)

    def __add__(self, other):
        """
        Returns a new :class:`LabelledCollection` as the union of this collection with another collection

        :param other: another :class:`LabelledCollection`
        :return: a :class:`LabelledCollection` representing the union of both collections
        """
        if other is None:
            return self
        elif issparse(self.instances) and issparse(other.instances):
            join_instances = vstack([self.instances, other.instances])
        elif isinstance(self.instances, list) and isinstance(other.instances, list):
            join_instances = self.instances + other.instances
        elif isinstance(self.instances, np.ndarray) and isinstance(other.instances, np.ndarray):
            join_instances = np.concatenate([self.instances, other.instances])
        else:
            raise NotImplementedError('unsupported operation for collection types')
        labels = np.concatenate([self.labels, other.labels])
        return LabelledCollection(join_instances, labels)

    @property
    def Xy(self):
        """
        Gets the instances and labels. This is useful when working with `sklearn` estimators, e.g.:

        >>> svm = LinearSVC().fit(*my_collection.Xy)

        :return: a tuple `(instances, labels)` from this collection
        """
        return self.instances, self.labels

    def stats(self, show=True):
        """
        Returns (and eventually prints) a dictionary with some stats of this collection. E.g.,:

        >>> data = qp.datasets.fetch_reviews('kindle', tfidf=True, min_df=5)
        >>> data.training.stats()
        >>> #instances=3821, type=<class 'scipy.sparse.csr.csr_matrix'>, #features=4403, #classes=[0 1], prevs=[0.081, 0.919]

        :param show: if set to True (default), prints the stats in standard output
        :return: a dictionary containing some stats of this collection. Keys include `#instances` (the number of
            instances), `type` (the type representing the instances), `#features` (the number of features, if the
            instances are in array-like format), `#classes` (the classes of the collection), `prevs` (the prevalence
            values for each class)
        """
        ninstances = len(self)
        instance_type = type(self.instances[0])
        if instance_type == list:
            nfeats = len(self.instances[0])
        elif instance_type == np.ndarray or issparse(self.instances):
            nfeats = self.instances.shape[1]
        else:
            nfeats = '?'
        stats_ = {'instances': ninstances,
                  'type': instance_type,
                  'features': nfeats,
                  'classes': self.classes_,
                  'prevs': strprev(self.prevalence())}
        if show:
            print(f'#instances={stats_["instances"]}, type={stats_["type"]}, #features={stats_["features"]}, '
                  f'#classes={stats_["classes"]}, prevs={stats_["prevs"]}')
        return stats_

    def kFCV(self, nfolds=5, nrepeats=1, random_state=0):
        """
        Generator of stratified folds to be used in k-fold cross validation.

        :param nfolds: integer (default 5), the number of folds to generate
        :param nrepeats: integer (default 1), the number of rounds of k-fold cross validation to run
        :param random_state: integer (default 0), guarantees that the folds generated are reproducible
        :return: yields `nfolds * nrepeats` folds for k-fold cross validation
        """
        kf = RepeatedStratifiedKFold(n_splits=nfolds, n_repeats=nrepeats, random_state=random_state)
        for train_index, test_index in kf.split(*self.Xy):
            train = self.sampling_from_index(train_index)
            test = self.sampling_from_index(test_index)
            yield train, test


class Dataset:
    """
    Abstraction of training and test :class:`LabelledCollection` objects.

    :param training: a :class:`LabelledCollection` instance
    :param test: a :class:`LabelledCollection` instance
    :param vocabulary: if indicated, is a dictionary of the terms used in this textual dataset
    :param name: a string representing the name of the dataset
    """

    def __init__(self, training: LabelledCollection, test: LabelledCollection, vocabulary: dict = None, name=''):
        assert set(training.classes_) == set(test.classes_), 'incompatible labels in training and test collections'
        self.training = training
        self.test = test
        self.vocabulary = vocabulary
        self.name = name

    @classmethod
    def SplitStratified(cls, collection: LabelledCollection, train_size=0.6):
        """
        Generates a :class:`Dataset` from a stratified split of a :class:`LabelledCollection` instance.
        See :meth:`LabelledCollection.split_stratified`

        :param collection: :class:`LabelledCollection`
        :param train_size: the proportion of training documents (the rest conforms the test split)
        :return: an instance of :class:`Dataset`
        """
        return Dataset(*collection.split_stratified(train_prop=train_size))

    @property
    def classes_(self):
        """
        The classes according to which the training collection is labelled

        :return: The classes according to which the training collection is labelled
        """
        return self.training.classes_

    @property
    def n_classes(self):
        """
        The number of classes according to which the training collection is labelled

        :return: integer
        """
        return self.training.n_classes

    @property
    def binary(self):
        """
        Returns True if the training collection is labelled according to two classes

        :return: boolean
        """
        return self.training.binary

    @classmethod
    def load(cls, train_path, test_path, loader_func: callable, classes=None, **loader_kwargs):
        """
        Loads a training and a test labelled set of data and convert it into a :class:`Dataset` instance.
        The function in charge of reading the instances must be specified. This function can be a custom one, or any of
        the reading functions defined in :mod:`quapy.data.reader` module.

        :param train_path: string, the path to the file containing the training instances
        :param test_path: string, the path to the file containing the test instances
        :param loader_func: a custom function that implements the data loader and returns a tuple with instances and
            labels
        :param classes: array-like, the classes according to which the instances are labelled
        :param loader_kwargs: any argument that the `loader_func` function needs in order to read the instances.
            See :meth:`LabelledCollection.load` for further details.
        :return: a :class:`Dataset` object
        """

        training = LabelledCollection.load(train_path, loader_func, classes, **loader_kwargs)
        test = LabelledCollection.load(test_path, loader_func, classes, **loader_kwargs)
        return Dataset(training, test)

    @property
    def vocabulary_size(self):
        """
        If the dataset is textual, and the vocabulary was indicated, returns the size of the vocabulary

        :return: integer
        """
        return len(self.vocabulary)

    def stats(self, show):
        """
        Returns (and eventually prints) a dictionary with some stats of this dataset. E.g.,:

        >>> data = qp.datasets.fetch_reviews('kindle', tfidf=True, min_df=5)
        >>> data.stats()
        >>> Dataset=kindle #tr-instances=3821, #te-instances=21591, type=<class 'scipy.sparse.csr.csr_matrix'>, #features=4403, #classes=[0 1], tr-prevs=[0.081, 0.919], te-prevs=[0.063, 0.937]

        :param show: if set to True (default), prints the stats in standard output
        :return: a dictionary containing some stats of this collection for the training and test collections. The keys
            are `train` and `test`, and point to dedicated dictionaries of stats, for each collection, with keys
            `#instances` (the number of instances), `type` (the type representing the instances),
            `#features` (the number of features, if the instances are in array-like format), `#classes` (the classes of
            the collection), `prevs` (the prevalence values for each class)
        """
        tr_stats = self.training.stats(show=False)
        te_stats = self.test.stats(show=False)
        if show:
            print(f'Dataset={self.name} #tr-instances={tr_stats["instances"]}, #te-instances={te_stats["instances"]}, '
                  f'type={tr_stats["type"]}, #features={tr_stats["features"]}, #classes={tr_stats["classes"]}, '
                  f'tr-prevs={tr_stats["prevs"]}, te-prevs={te_stats["prevs"]}')
        return {'train': tr_stats, 'test': te_stats}

    @classmethod
    def kFCV(cls, data: LabelledCollection, nfolds=5, nrepeats=1, random_state=0):
        """
        Generator of stratified folds to be used in k-fold cross validation. This function is only a wrapper around
        :meth:`LabelledCollection.kFCV` that returns :class:`Dataset` instances made of training and test folds.

        :param nfolds: integer (default 5), the number of folds to generate
        :param nrepeats: integer (default 1), the number of rounds of k-fold cross validation to run
        :param random_state: integer (default 0), guarantees that the folds generated are reproducible
        :return: yields `nfolds * nrepeats` folds for k-fold cross validation as instances of :class:`Dataset`
        """
        for i, (train, test) in enumerate(data.kFCV(nfolds=nfolds, nrepeats=nrepeats, random_state=random_state)):
            yield Dataset(train, test, name=f'fold {(i % nfolds) + 1}/{nfolds} (round={(i // nfolds) + 1})')


def isbinary(data):
    """
    Returns True if `data` is either a binary :class:`Dataset` or a binary :class:`LabelledCollection`

    :param data: a :class:`Dataset` or a :class:`LabelledCollection` object
    :return: True if labelled according to two classes
    """
    if isinstance(data, Dataset) or isinstance(data, LabelledCollection):
        return data.binary
    return False
