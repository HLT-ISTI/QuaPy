import contextlib
import itertools
import multiprocessing
import os
import pickle
import urllib
from pathlib import Path
import quapy as qp

import numpy as np
from joblib import Parallel, delayed


def get_parallel_slices(n_tasks, n_jobs=-1):
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    batch = int(n_tasks / n_jobs)
    remainder = n_tasks % n_jobs
    return [slice(job * batch, (job + 1) * batch + (remainder if job == n_jobs - 1 else 0)) for job in
            range(n_jobs)]


def map_parallel(func, args, n_jobs):
    """
    Applies func to n_jobs slices of args. E.g., if args is an array of 99 items and n_jobs=2, then
    func is applied in two parallel processes to args[0:50] and to args[50:99]
    """
    args = np.asarray(args)
    slices = get_parallel_slices(len(args), n_jobs)
    results = Parallel(n_jobs=n_jobs)(
        delayed(func)(args[slice_i]) for slice_i in slices
    )
    return list(itertools.chain.from_iterable(results))


def parallel(func, args, n_jobs):
    """
    A wrapper of multiprocessing:
    Parallel(n_jobs=n_jobs)(
        delayed(func)(args_i) for args_i in args
    )
    that takes the quapy.environ variable as input silently
    """
    def func_dec(environ, *args):
        qp.environ = environ
        return func(*args)
    return Parallel(n_jobs=n_jobs)(
        delayed(func_dec)(qp.environ, args_i) for args_i in args
    )



@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def download_file(url, archive_filename):
    def progress(blocknum, bs, size):
        total_sz_mb = '%.2f MB' % (size / 1e6)
        current_sz_mb = '%.2f MB' % ((blocknum * bs) / 1e6)
        print('\rdownloaded %s / %s' % (current_sz_mb, total_sz_mb), end='')
    print("Downloading %s" % url)
    urllib.request.urlretrieve(url, filename=archive_filename, reporthook=progress)
    print("")


def download_file_if_not_exists(url, archive_path):
    if os.path.exists(archive_path):
        return
    create_if_not_exist(os.path.dirname(archive_path))
    download_file(url,archive_path)


def create_if_not_exist(path):
    os.makedirs(path, exist_ok=True)


def get_quapy_home():
    home = os.path.join(str(Path.home()), 'quapy_data')
    os.makedirs(home, exist_ok=True)
    return home


def create_parent_dir(path):
    os.makedirs(Path(path).parent, exist_ok=True)


def pickled_resource(pickle_path:str, generation_func:callable, *args):
    if pickle_path is None:
        return generation_func(*args)
    else:
        if os.path.exists(pickle_path):
            return pickle.load(open(pickle_path, 'rb'))
        else:
            instance = generation_func(*args)
            os.makedirs(str(Path(pickle_path).parent), exist_ok=True)
            pickle.dump(instance, open(pickle_path, 'wb'), pickle.HIGHEST_PROTOCOL)
            return instance


class EarlyStop:

    def __init__(self, patience, lower_is_better=True):
        self.PATIENCE_LIMIT = patience
        self.better = lambda a,b: a<b if lower_is_better else a>b
        self.patience = patience
        self.best_score = None
        self.best_epoch = None
        self.STOP = False
        self.IMPROVED = False

    def __call__(self, watch_score, epoch):
        self.IMPROVED = (self.best_score is None or self.better(watch_score, self.best_score))
        if self.IMPROVED:
            self.best_score = watch_score
            self.best_epoch = epoch
            self.patience = self.PATIENCE_LIMIT
        else:
            self.patience -= 1
            if self.patience <= 0:
                self.STOP = True