import numpy as np
from scipy.sparse import issparse, dok_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from quapy.functional import artificial_prevalence_sampling
from scipy.sparse import vstack


class LabelledCollection:

    def __init__(self, instances, labels, n_classes=None):
        self.instances = instances if issparse(instances) else np.asarray(instances)
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

    @classmethod
    def load_dataset(cls, train_path, test_path):
        training = cls.load(train_path)
        test = cls.load(test_path)
        return Dataset(training, test)

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
        return self.n_classes==2

    def sampling_index(self, size, *prevs, shuffle=True):
        if len(prevs) == self.n_classes-1:
            prevs = prevs + (1-sum(prevs),)
        assert len(prevs) == self.n_classes, 'unexpected number of prevalences'
        assert sum(prevs) == 1, f'prevalences ({prevs}) out of range (sum={sum(prevs)})'

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

    def sampling(self, size, *prevs, shuffle=True):
        index = self.sampling_index(size, *prevs, shuffle=shuffle)
        return self.sampling_from_index(index)

    def sampling_from_index(self, index):
        documents = self.instances[index]
        labels = self.labels[index]
        return LabelledCollection(documents, labels, n_classes=self.n_classes)

    def split_stratified(self, train_prop=0.6):
        tr_docs, te_docs, tr_labels, te_labels = \
            train_test_split(self.instances, self.labels, train_size=train_prop, stratify=self.labels)
        return LabelledCollection(tr_docs, tr_labels), LabelledCollection(te_docs, te_labels)

    def artificial_sampling_generator(self, sample_size, n_prevalences=101, repeats=1):
        dimensions=self.n_classes
        for prevs in artificial_prevalence_sampling(dimensions, n_prevalences, repeats):
            yield self.sampling(sample_size, *prevs)

    def __add__(self, other):
        if issparse(self.instances) and issparse(other.documents):
            docs = vstack([self.instances, other.documents])
        elif isinstance(self.instances, list) and isinstance(other.documents, list):
            docs = self.instances + other.documents
        else:
            raise NotImplementedError('unsupported operation for collection types')
        labels = np.concatenate([self.labels, other.labels])
        return LabelledCollection(docs, labels)



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




