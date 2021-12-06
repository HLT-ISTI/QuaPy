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
    Transforms a :class:`quapy.data.base.Dataset` of textual instances into a :class:`quapy.data.base.Dataset` of
    tfidf weighted sparse vectors

    :param dataset: a :class:`quapy.data.base.Dataset` where the instances of training and test collections are
        lists of str
    :param min_df: minimum number of occurrences for a word to be considered as part of the vocabulary (default 3)
    :param sublinear_tf: whether or not to apply the log scalling to the tf counters (default True)
    :param inplace: whether or not to apply the transformation inplace (True), or to a new copy (False, default)
    :param kwargs: the rest of parameters of the transformation (as for sklearn's
        `TfidfVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>`_)
    :return: a new :class:`quapy.data.base.Dataset` in `csr_matrix` format (if inplace=False) or a reference to the
        current Dataset (if inplace=True) where the instances are stored in a `csr_matrix` of real-valued tfidf scores
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
    Reduces the dimensionality of the instances, represented as a `csr_matrix` (or any subtype of
    `scipy.sparse.spmatrix`), of training and test documents by removing the columns of words which are not present
    in at least `min_df` instances in the training set

    :param dataset: a :class:`quapy.data.base.Dataset` in which instances are represented in sparse format (any
        subtype of scipy.sparse.spmatrix)
    :param min_df: integer, minimum number of instances below which the columns are removed
    :param inplace: whether or not to apply the transformation inplace (True), or to a new copy (False, default)
    :return: a new :class:`quapy.data.base.Dataset` (if inplace=False) or a reference to the current
        :class:`quapy.data.base.Dataset` (inplace=True) where the dimensions corresponding to infrequent terms
        in the training set have been removed
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


def standardize(dataset: Dataset, inplace=False):
    """
    Standardizes the real-valued columns of a :class:`quapy.data.base.Dataset`.
    Standardization, aka z-scoring, of a variable `X` comes down to subtracting the average and normalizing by the
    standard deviation.

    :param dataset: a :class:`quapy.data.base.Dataset` object
    :param inplace: set to True if the transformation is to be applied inplace, or to False (default) if a new
        :class:`quapy.data.base.Dataset` is to be returned
    :return:
    """
    s = StandardScaler(copy=not inplace)
    training = s.fit_transform(dataset.training.instances)
    test = s.transform(dataset.test.instances)
    if inplace:
        return dataset
    else:
        return Dataset(training, test, dataset.vocabulary, dataset.name)


def index(dataset: Dataset, min_df=5, inplace=False, **kwargs):
    """
    Indexes the tokens of a textual :class:`quapy.data.base.Dataset` of string documents.
    To index a document means to replace each different token by a unique numerical index.
    Rare words (i.e., words occurring less than `min_df` times) are replaced by a special token `UNK`

    :param dataset: a :class:`quapy.data.base.Dataset` object where the instances of training and test documents
        are lists of str
    :param min_df: minimum number of occurrences below which the term is replaced by a `UNK` index
    :param inplace: whether or not to apply the transformation inplace (True), or to a new copy (False, default)
    :param kwargs: the rest of parameters of the transformation (as for sklearn's
    `CountVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>_`)
    :return: a new :class:`quapy.data.base.Dataset` (if inplace=False) or a reference to the current
        :class:`quapy.data.base.Dataset` (inplace=True) consisting of lists of integer values representing indices.
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
    """
    This class implements a sklearn's-style transformer that indexes text as numerical ids for the tokens it
    contains, and that would be generated by sklearn's
    `CountVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_

    :param kwargs: keyworded arguments from `CountVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_
    """

    def __init__(self, **kwargs):
        self.vect = CountVectorizer(**kwargs)
        self.unk = -1  # a valid index is assigned after fit
        self.pad = -2  # a valid index is assigned after fit

    def fit(self, X):
        """
        Fits the transformer, i.e., decides on the vocabulary, given a list of strings.

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
        """
        Transforms the strings in `X` as lists of numerical ids

        :param X: a list of strings
        :param n_jobs: the number of parallel workers to carry out this task
        :return: a `np.ndarray` of numerical ids
        """
        # given the number of tasks and the number of jobs, generates the slices for the parallel processes
        assert self.unk != -1, 'transform called before fit'
        indexed = map_parallel(func=self._index, args=X, n_jobs=n_jobs)
        return np.asarray(indexed)

    def _index(self, documents):
        vocab = self.vocabulary_.copy()
        return [[vocab.prevalence(word, self.unk) for word in self.analyzer(doc)] for doc in tqdm(documents, 'indexing')]

    def fit_transform(self, X, n_jobs=-1):
        """
        Fits the transform on `X` and transforms it.

        :param X: a list of strings
        :param n_jobs: the number of parallel workers to carry out this task
        :return: a `np.ndarray` of numerical ids
        """
        return self.fit(X).transform(X, n_jobs=n_jobs)

    def vocabulary_size(self):
        """
        Gets the length of the vocabulary according to which the document tokens have been indexed

        :return: integer
        """
        return len(self.vocabulary_)

    def add_word(self, word, id=None, nogaps=True):
        """
        Adds a new token (regardless of whether it has been found in the text or not), with dedicated id.
        Useful to define special tokens for codifying unknown words, or padding tokens.

        :param word: string, surface form of the token
        :param id: integer, numerical value to assign to the token (leave as None for indicating the next valid id,
            default)
        :param nogaps: if set to True (default) asserts that the id indicated leads to no numerical gaps with
            precedent ids stored so far
        :return: integer, the numerical id for the new token
        """
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

