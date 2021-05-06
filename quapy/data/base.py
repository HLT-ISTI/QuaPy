import numpy as np
from scipy.sparse import issparse
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

from quapy.functional import artificial_prevalence_sampling, strprev


class LabelledCollection:
    '''
    A LabelledCollection is a set of objects each with a label associated to it.
    '''

    def __init__(self, instances, labels, classes_=None):
        """
        :param instances: list of objects
        :param labels: list of labels, same length of instances
        :param classes_: optional, list of classes from which labels are taken. When used, must contain the set of values used in labels.
        """
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
                raise ValueError('labels contains values not included in classes_')
        self.index = {class_: np.arange(n_docs)[self.labels == class_] for class_ in self.classes_}

    @classmethod
    def load(cls, path: str, loader_func: callable):
        return LabelledCollection(*loader_func(path))

    def __len__(self):
        return self.instances.shape[0]

    def prevalence(self):
        return self.counts() / len(self)

    def counts(self):
        return np.asarray([len(self.index[class_]) for class_ in self.classes_])

    @property
    def n_classes(self):
        return len(self.classes_)

    @property
    def binary(self):
        return self.n_classes == 2

    def sampling_index(self, size, *prevs, shuffle=True):
        if len(prevs) == 0:  # no prevalence was indicated; returns an index for uniform sampling
            return np.random.choice(len(self), size, replace=False)
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
        return LabelledCollection(documents, labels, classes_=self.classes_)

    def split_stratified(self, train_prop=0.6, random_state=None):
        # with temp_seed(42):
        tr_docs, te_docs, tr_labels, te_labels = \
            train_test_split(self.instances, self.labels, train_size=train_prop, stratify=self.labels,
                             random_state=random_state)
        return LabelledCollection(tr_docs, tr_labels), LabelledCollection(te_docs, te_labels)

    def artificial_sampling_generator(self, sample_size, n_prevalences=101, repeats=1):
        dimensions = self.n_classes
        for prevs in artificial_prevalence_sampling(dimensions, n_prevalences, repeats):
            yield self.sampling(sample_size, *prevs)

    def artificial_sampling_index_generator(self, sample_size, n_prevalences=101, repeats=1):
        dimensions = self.n_classes
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

    @property
    def Xy(self):
        return self.instances, self.labels

    def stats(self, show=True):
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
        kf = RepeatedStratifiedKFold(n_splits=nfolds, n_repeats=nrepeats, random_state=random_state)
        for train_index, test_index in kf.split(*self.Xy):
            train = self.sampling_from_index(train_index)
            test = self.sampling_from_index(test_index)
            yield train, test


class Dataset:

    def __init__(self, training: LabelledCollection, test: LabelledCollection, vocabulary: dict = None, name=''):
        assert set(training.classes_) == set(test.classes_), 'incompatible labels in training and test collections'
        self.training = training
        self.test = test
        self.vocabulary = vocabulary
        self.name = name

    @classmethod
    def SplitStratified(cls, collection: LabelledCollection, train_size=0.6):
        return Dataset(*collection.split_stratified(train_prop=train_size))

    @property
    def classes_(self):
        return self.training.classes_

    @property
    def n_classes(self):
        return self.training.n_classes

    @property
    def binary(self):
        return self.training.binary

    @classmethod
    def load(cls, train_path, test_path, loader_func: callable):
        training = LabelledCollection.load(train_path, loader_func)
        test = LabelledCollection.load(test_path, loader_func)
        return Dataset(training, test)

    @property
    def vocabulary_size(self):
        return len(self.vocabulary)

    def stats(self):
        tr_stats = self.training.stats(show=False)
        te_stats = self.test.stats(show=False)
        print(f'Dataset={self.name} #tr-instances={tr_stats["instances"]}, #te-instances={te_stats["instances"]}, '
              f'type={tr_stats["type"]}, #features={tr_stats["features"]}, #classes={tr_stats["classes"]}, '
              f'tr-prevs={tr_stats["prevs"]}, te-prevs={te_stats["prevs"]}')
        return {'train': tr_stats, 'test': te_stats}

    @classmethod
    def kFCV(cls, data: LabelledCollection, nfolds=5, nrepeats=1, random_state=0):
        for i, (train, test) in enumerate(data.kFCV(nfolds=nfolds, nrepeats=nrepeats, random_state=random_state)):
            yield Dataset(train, test, name=f'fold {(i % nfolds) + 1}/{nfolds} (round={(i // nfolds) + 1})')


def isbinary(data):
    if isinstance(data, Dataset) or isinstance(data, LabelledCollection):
        return data.binary
    return False
