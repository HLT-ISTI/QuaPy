import numpy as np
from scipy.sparse import dok_matrix
from tqdm import tqdm


def from_text(path, encoding='utf-8'):
    """
    Reas a labelled colletion of documents.
    File fomart <0 or 1>\t<document>\n
    :param path: path to the labelled collection
    :return: a list of sentences, and a list of labels
    """
    all_sentences, all_labels = [], []
    for line in tqdm(open(path, 'rt', encoding=encoding).readlines(), f'loading {path}'):
        line = line.strip()
        if line:
            label, sentence = line.split('\t')
            sentence = sentence.strip()
            label = int(label)
            if sentence:
                all_sentences.append(sentence)
                all_labels.append(label)
    return all_sentences, all_labels


def from_sparse(path):
    """
    Reads a labelled collection of real-valued instances expressed in sparse format
    File format <-1 or 0 or 1>[\s col(int):val(float)]\n
    :param path: path to the labelled collection
    :return: a csr_matrix containing the instances (rows), and a ndarray containing the labels
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
    :return: a ndarray for the labels and a ndarray (float) for the covariates
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
    E.g., y=['B', 'B', 'A', 'C'] -> [1,1,0,2], ['A','B','C']
    :param y: the list or array of original labels
    :return: a ndarray (int) of class indexes, and a ndarray of classnames corresponding to the indexes.
    """
    classnames = sorted(np.unique(y))
    label2index = {label: index for index, label in enumerate(classnames)}
    indexed = np.empty(y.shape, dtype=np.int)
    for label in classnames:
        indexed[y==label] = label2index[label]
    return indexed, classnames


def binarize(y, pos_class):
    y = np.asarray(y)
    ybin = np.zeros(y.shape, dtype=np.int)
    ybin[y == pos_class] = 1
    return ybin

