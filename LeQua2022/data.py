import quapy as qp
import numpy as np


def load_binary_raw_document(path):
    documents, labels = qp.data.from_text(path, verbose=0, class2int=True)
    labels = np.asarray(labels)
    labels[np.logical_or(labels == 1, labels == 2)] = 0
    labels[np.logical_or(labels == 4, labels == 5)] = 1
    return documents, labels


def load_multiclass_raw_document(path):
    return qp.data.from_text(path, verbose=0, class2int=False)


