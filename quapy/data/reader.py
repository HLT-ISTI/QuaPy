import numpy as np
from scipy.sparse import dok_matrix
from tqdm import tqdm


def from_text(path, encoding='utf-8', verbose=1, class2int=True):
    """
    Reads a labelled colletion of documents.
    File fomart <0 or 1>\t<document>\n

    :param path: path to the labelled collection
    :param encoding: the text encoding used to open the file
    :param verbose: if >0 (default) shows some progress information in standard output
    :return: a list of sentences, and a list of labels
    """
    all_sentences, all_labels = [], []
    if verbose>0:
        file = tqdm(open(path, 'rt', encoding=encoding).readlines(), f'loading {path}')
    else:
        file = open(path, 'rt', encoding=encoding).readlines()
    for line in file:
        line = line.strip()
        if line:
            try:
                label, sentence = line.split('\t')
                sentence = sentence.strip()
                if class2int:
                    label = int(label)
                if sentence:
                    all_sentences.append(sentence)
                    all_labels.append(label)
            except ValueError:
                print(f'format error in {line}')
    return all_sentences, all_labels


def from_sparse(path):
    """
    Reads a labelled collection of real-valued instances expressed in sparse format
    File format <-1 or 0 or 1>[\s col(int):val(float)]\n

    :param path: path to the labelled collection
    :return: a `csr_matrix` containing the instances (rows), and a ndarray containing the labels
    """

    def split_col_val(col_val):
        col, val = col_val.split(':')
        col, val = int(col) - 1, float(val)
        return col, val

    all_documents, all_labels = [], []
    max_col = 0
    for line in tqdm(open(path, 'rt').readlines(), f'loading {path}'):
        parts = line.strip().split()
        if parts:
            all_labels.append(int(parts[0]))
            cols, vals = zip(*[split_col_val(col_val) for col_val in parts[1:]])
            cols, vals = np.asarray(cols), np.asarray(vals)
            max_col = max(max_col, cols.max())
            all_documents.append((cols, vals))
    n_docs = len(all_labels)
    X = dok_matrix((n_docs, max_col + 1), dtype=float)
    for i, (cols, vals) in tqdm(enumerate(all_documents), total=len(all_documents),
                                desc=f'\-- filling matrix of shape {X.shape}'):
        X[i, cols] = vals
    X = X.tocsr()
    y = np.asarray(all_labels) + 1
    return X, y


def from_csv(path, encoding='utf-8'):
    """
    Reads a csv file in which columns are separated by ','.
    File format <label>,<feat1>,<feat2>,...,<featn>\n

    :param path: path to the csv file
    :param encoding: the text encoding used to open the file
    :return: a np.ndarray for the labels and a ndarray (float) for the covariates
    """

    X, y = [], []
    for instance in tqdm(open(path, 'rt', encoding=encoding).readlines(), desc=f'reading {path}'):
        yi, *xi = instance.strip().split(',')
        X.append(list(map(float,xi)))
        y.append(yi)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


def reindex_labels(y):
    """
    Re-indexes a list of labels as a list of indexes, and returns the classnames corresponding to the indexes.
    E.g.:

    >>> reindex_labels(['B', 'B', 'A', 'C'])
    >>> (array([1, 1, 0, 2]), array(['A', 'B', 'C'], dtype='<U1'))

    :param y: the list or array of original labels
    :return: a ndarray (int) of class indexes, and a ndarray of classnames corresponding to the indexes.
    """
    y = np.asarray(y)
    classnames = np.asarray(sorted(np.unique(y)))
    label2index = {label: index for index, label in enumerate(classnames)}
    indexed = np.empty(y.shape, dtype=np.int)
    for label in classnames:
        indexed[y==label] = label2index[label]
    return indexed, classnames


def binarize(y, pos_class):
    """
    Binarizes a categorical array-like collection of labels towards the positive class `pos_class`. E.g.,:

    >>> binarize([1, 2, 3, 1, 1, 0], pos_class=2)
    >>> array([0, 1, 0, 0, 0, 0])

    :param y: array-like of labels
    :param pos_class: integer, the positive class
    :return: a binary np.ndarray, in which values 1 corresponds to positions in whcih `y` had `pos_class` labels, and
        0 otherwise
    """
    y = np.asarray(y)
    ybin = np.zeros(y.shape, dtype=np.int)
    ybin[y == pos_class] = 1
    return ybin

