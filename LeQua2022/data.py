import quapy as qp
import numpy as np
import sklearn


# def load_binary_raw_document(path):
#     documents, labels = qp.data.from_text(path, verbose=0, class2int=True)
#     labels = np.asarray(labels)
#     labels[np.logical_or(labels == 1, labels == 2)] = 0
#     labels[np.logical_or(labels == 4, labels == 5)] = 1
#     return documents, labels


def load_multiclass_raw_document(path):
    return qp.data.from_text(path, verbose=0, class2int=False)


def load_binary_vectors(path, nF=None):
    return sklearn.datasets.load_svmlight_file(path, n_features=nF)


if __name__ == '__main__':
    X, y = load_binary_vectors('./data/T1A/public/training_vectors.txt')
    print(X.shape)
    print(y)


