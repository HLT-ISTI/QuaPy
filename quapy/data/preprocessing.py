import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import quapy as qp
from quapy.data.base import Dataset
from quapy.util import map_parallel
from .base import LabelledCollection


def text2tfidf(dataset:Dataset, min_df=3, sublinear_tf=True, inplace=False, **kwargs):
    """
    Transforms a Dataset of textual instances into a Dataset of tfidf weighted sparse vectors
    :param dataset: a Dataset where the instances are lists of str
    :param min_df: minimum number of occurrences for a word to be considered as part of the vocabulary
    :param sublinear_tf: whether or not to apply the log scalling to the tf counters
    :param inplace: whether or not to apply the transformation inplace, or to a new copy
    :param kwargs: the rest of parameters of the transformation (as for sklearn.feature_extraction.text.TfidfVectorizer)
    :return: a new Dataset in csr_matrix format (if inplace=False) or a reference to the current Dataset (inplace=True)
    where the instances are stored in a csr_matrix of real-valued tfidf scores
    """
    __check_type(dataset.training.instances, np.ndarray, str)
    __check_type(dataset.test.instances, np.ndarray, str)

    vectorizer = TfidfVectorizer(min_df=min_df, sublinear_tf=sublinear_tf, **kwargs)
    training_documents = vectorizer.fit_transform(dataset.training.instances)
    test_documents = vectorizer.transform(dataset.test.instances)

    if inplace:
        dataset.training = LabelledCollection(training_documents, dataset.training.labels, dataset.classes_)
        dataset.test = LabelledCollection(test_documents, dataset.test.labels, dataset.classes_)
        dataset.vocabulary = vectorizer.vocabulary_
        return dataset
    else:
        training = LabelledCollection(training_documents, dataset.training.labels.copy(), dataset.classes_)
        test = LabelledCollection(test_documents, dataset.test.labels.copy(), dataset.classes_)
        return Dataset(training, test, vectorizer.vocabulary_)


def reduce_columns(dataset: Dataset, min_df=5, inplace=False):
    """
    Reduces the dimensionality of the csr_matrix by removing the columns of words which are not present in at least
    _min_df_ instances
    :param dataset: a Dataset in sparse format (any subtype of scipy.sparse.spmatrix)
    :param min_df: minimum number of instances below which the columns are removed
    :param inplace: whether or not to apply the transformation inplace, or to a new copy
    :return: a new Dataset (if inplace=False) or a reference to the current Dataset (inplace=True)
    where the dimensions corresponding to infrequent instances have been removed
    """
    __check_type(dataset.training.instances, spmatrix)
    __check_type(dataset.test.instances, spmatrix)
    assert dataset.training.instances.shape[1] == dataset.test.instances.shape[1], 'unaligned vector spaces'

    def filter_by_occurrences(X, W):
        column_prevalence = np.asarray((X > 0).sum(axis=0)).flatten()
        take_columns = column_prevalence >= min_df
        X = X[:, take_columns]
        W = W[:, take_columns]
        return X, W

    Xtr, Xte = filter_by_occurrences(dataset.training.instances, dataset.test.instances)
    if inplace:
        dataset.training.instances = Xtr
        dataset.test.instances = Xte
        return dataset
    else:
        training = LabelledCollection(Xtr, dataset.training.labels.copy(), dataset.classes_)
        test = LabelledCollection(Xte, dataset.test.labels.copy(), dataset.classes_)
        return Dataset(training, test)


def standardize(dataset: Dataset, inplace=True):
    s = StandardScaler(copy=not inplace)
    training = s.fit_transform(dataset.training.instances)
    test = s.transform(dataset.test.instances)
    if inplace:
        return dataset
    else:
        return Dataset(training, test, dataset.vocabulary, dataset.name)


def index(dataset: Dataset, min_df=5, inplace=False, **kwargs):
    """
    Indexes a dataset of strings. To index a document means to replace each different token by a unique numerical index.
    Rare words (i.e., words occurring less than _min_df_ times) are replaced by a special token UNK
    :param dataset: a Dataset where the instances are lists of str
    :param min_df: minimum number of instances below which the term is replaced by a UNK index
    :param inplace: whether or not to apply the transformation inplace, or to a new copy
    :param kwargs: the rest of parameters of the transformation (as for sklearn.feature_extraction.text.CountVectorizer)
    :return: a new Dataset (if inplace=False) or a reference to the current Dataset (inplace=True)
    consisting of lists of integer values representing indices.
    """
    __check_type(dataset.training.instances, np.ndarray, str)
    __check_type(dataset.test.instances, np.ndarray, str)

    indexer = IndexTransformer(min_df=min_df, **kwargs)
    training_index = indexer.fit_transform(dataset.training.instances)
    test_index = indexer.transform(dataset.test.instances)

    if inplace:
        dataset.training = LabelledCollection(training_index, dataset.training.labels, dataset.classes_)
        dataset.test = LabelledCollection(test_index, dataset.test.labels, dataset.classes_)
        dataset.vocabulary = indexer.vocabulary_
        return dataset
    else:
        training = LabelledCollection(training_index, dataset.training.labels.copy(), dataset.classes_)
        test = LabelledCollection(test_index, dataset.test.labels.copy(), dataset.classes_)
        return Dataset(training, test, indexer.vocabulary_)


def __check_type(container, container_type=None, element_type=None):
    if container_type:
        assert isinstance(container, container_type), \
            f'unexpected type of container (expected {container_type}, found {type(container)})'
    if element_type:
        assert isinstance(container[0], element_type), \
            f'unexpected type of element (expected {container_type}, found {type(container)})'


class IndexTransformer:

    def __init__(self, **kwargs):
        """
        :param kwargs: keyworded arguments from _sklearn.feature_extraction.text.CountVectorizer_
        """
        self.vect = CountVectorizer(**kwargs)
        self.unk = -1  # a valid index is assigned after fit
        self.pad = -2  # a valid index is assigned after fit

    def fit(self, X):
        """
        :param X: a list of strings
        :return: self
        """
        self.vect.fit(X)
        self.analyzer = self.vect.build_analyzer()
        self.vocabulary_ = self.vect.vocabulary_
        self.unk = self.add_word(qp.environ['UNK_TOKEN'], qp.environ['UNK_INDEX'])
        self.pad = self.add_word(qp.environ['PAD_TOKEN'], qp.environ['PAD_INDEX'])
        return self

    def transform(self, X, n_jobs=-1):
        # given the number of tasks and the number of jobs, generates the slices for the parallel processes
        assert self.unk != -1, 'transform called before fit'
        indexed = map_parallel(func=self.index, args=X, n_jobs=n_jobs)
        return np.asarray(indexed)

    def index(self, documents):
        vocab = self.vocabulary_.copy()
        return [[vocab.get(word, self.unk) for word in self.analyzer(doc)] for doc in tqdm(documents, 'indexing')]

    def fit_transform(self, X, n_jobs=-1):
        return self.fit(X).transform(X, n_jobs=n_jobs)

    def vocabulary_size(self):
        return len(self.vocabulary_)

    def add_word(self, word, id=None, nogaps=True):
        if word in self.vocabulary_:
            raise ValueError(f'word {word} already in dictionary')
        if id is None:
            # add the word with the next id
            self.vocabulary_[word] = len(self.vocabulary_)
        else:
            id2word = {id_:word_ for word_, id_ in self.vocabulary_.items()}
            if id in id2word:
                old_word = id2word[id]
                self.vocabulary_[word] = id
                del self.vocabulary_[old_word]
                self.add_word(old_word)
            elif nogaps:
                if id > self.vocabulary_size()+1:
                    raise ValueError(f'word {word} added with id {id}, while the current vocabulary size '
                                     f'is of {self.vocabulary_size()}, and id gaps are not allowed')
        return self.vocabulary_[word]

