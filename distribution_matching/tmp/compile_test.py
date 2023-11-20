from time import time
from sklearn.metrics.pairwise import rbf_kernel
import torch
import numpy as np


def rbf(X, Y):
    return X @ Y.T


@torch.compile
def rbf_comp(X, Y):
    return X @ Y.T


def measure_time(X, Y, func):
    tinit = time()
    func(X, Y)
    tend = time()
    print(f'took {tend-tinit:.3}s')


X = np.random.rand(1000, 100)
Y = np.random.rand(1000, 100)

measure_time(X, Y, rbf)
measure_time(X, Y, rbf_comp)
