import numpy as np
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from quapy.functional import artificial_prevalence_sampling
from scipy.sparse import vstack

from util import temp_seed


class LabelledCollection:

    def __init__(self, instances, labels, n_classes=None):
        if issparse(instances):
            self.instances = instances
        elif isinstance(instances, list) and len(instances)>0 and isinstance(instances[0], str):
            # lists of strings occupy too much as ndarrays (although python-objects add a heavy overload)
            self.instances = np.asarray(instances, dtype=object)
        else:
            self.instances = np.asarray(instances)
        self.labels = np.asarray(labels, dtype=int)
        n_docs = len(self)
        if n_classes is None:
            self.classes_ = np.unique(self.labels)
            self.classes_.sort()
        else:
            self.classes_ = np.arange(n_classes)
        self.index = {class_i: np.arange(n_docs)[self.labels == class_i] for class_i in self.classes_}

    @classmethod
    def load(cls, path:str, loader_func:callable):
        return LabelledCollection(*loader_func(path))

    def __len__(self):
        return self.instances.shape[0]

    def prevalence(self):
        return self.counts()/len(self)

    def counts(self):
        return np.asarray([len(self.index[ci]) for ci in self.classes_])

    @property
    def n_classes(self):
        return len(self.classes_)

    @property
    def binary(self):
        return self.n_classes == 2

    def sampling_index(self, size, *prevs, shuffle=True):
        if len(prevs) == 0:  # no prevalence was indicated; returns an index for uniform sampling
            return np.random.choice(len(self), size, replace=False)
        if len(prevs) == self.n_classes-1:
            prevs = prevs + (1-sum(prevs),)
        assert len(prevs) == self.n_classes, 'unexpected number of prevalences'
        assert sum(prevs) == 1, f'prevalences ({prevs}) wrong range (sum={sum(prevs)})'

        taken = 0
        indexes_sample = []
        for i, class_i in enumerate(self.classes_):
            if i == self.n_classes-1:
                n_requested = size - taken
            else:
                n_requested = int(size * prevs[i])

            n_candidates = len(self.index[class_i])
            index_sample = self.index[class_i][
                np.random.choice(n_candidates, size=n_requested, replace=(n_requested > n_candidates))
            ] if n_requested > 0 else []

            indexes_sample.append(index_sample)
            taken += n_requested

        indexes_sample = np.concatenate(indexes_sample).astype(int)

        if shuffle:
            indexes_sample = np.random.permutation(indexes_sample)

        return indexes_sample

    # def uniform_sampling_index(self, size):
    #     return np.random.choice(len(self), size, replace=False)

    # def uniform_sampling(self, size):
    #     unif_index = self.uniform_sampling_index(size)
    #     return self.sampling_from_index(unif_index)

    def sampling(self, size, *prevs, shuffle=True):
        prev_index = self.sampling_index(size, *prevs, shuffle=shuffle)
        return self.sampling_from_index(prev_index)

    def sampling_from_index(self, index):
        documents = self.instances[index]
        labels = self.labels[index]
        return LabelledCollection(documents, labels, n_classes=self.n_classes)

    def split_stratified(self, train_prop=0.6):
        # with temp_seed(42):
        tr_docs, te_docs, tr_labels, te_labels = \
            train_test_split(self.instances, self.labels, train_size=train_prop, stratify=self.labels)
        return LabelledCollection(tr_docs, tr_labels), LabelledCollection(te_docs, te_labels)

    def artificial_sampling_generator(self, sample_size, n_prevalences=101, repeats=1):
        dimensions=self.n_classes
        for prevs in artificial_prevalence_sampling(dimensions, n_prevalences, repeats):
            yield self.sampling(sample_size, *prevs)

    def artificial_sampling_index_generator(self, sample_size, n_prevalences=101, repeats=1):
        dimensions=self.n_classes
        for prevs in artificial_prevalence_sampling(dimensions, n_prevalences, repeats):
            yield self.sampling_index(sample_size, *prevs)

    def natural_sampling_generator(self, sample_size, repeats=100):
        for _ in range(repeats):
            yield self.uniform_sampling(sample_size)

    def natural_sampling_index_generator(self, sample_size, repeats=100):
        for _ in range(repeats):
            yield self.uniform_sampling_index(sample_size)

    def __add__(self, other):
        if issparse(self.instances) and issparse(other.instances):
            join_instances = vstack([self.instances, other.instances])
        elif isinstance(self.instances, list) and isinstance(other.instances, list):
            join_instances = self.instances + other.instances
        elif isinstance(self.instances, np.ndarray) and isinstance(other.instances, np.ndarray):
            join_instances = np.concatenate([self.instances, other.instances])
        else:
            raise NotImplementedError('unsupported operation for collection types')
        labels = np.concatenate([self.labels, other.labels])
        return LabelledCollection(join_instances, labels)



class Dataset:

    def __init__(self, training: LabelledCollection, test: LabelledCollection, vocabulary: dict = None):
        assert training.n_classes == test.n_classes, 'incompatible labels in training and test collections'
        self.training = training
        self.test = test
        self.vocabulary = vocabulary

    @classmethod
    def SplitStratified(cls, collection: LabelledCollection, train_size=0.6):
        return Dataset(*collection.split_stratified(train_prop=train_size))

    @property
    def n_classes(self):
        return self.training.n_classes

    @property
    def binary(self):
        return self.training.binary

    @classmethod
    def load(cls, train_path, test_path, loader_func:callable):
        training = LabelledCollection.load(train_path, loader_func)
        test = LabelledCollection.load(test_path, loader_func)
        return Dataset(training, test)




