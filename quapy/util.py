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


def _get_parallel_slices(n_tasks, n_jobs=-1):
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    batch = int(n_tasks / n_jobs)
    remainder = n_tasks % n_jobs
    return [slice(job * batch, (job + 1) * batch + (remainder if job == n_jobs - 1 else 0)) for job in range(n_jobs)]


def map_parallel(func, args, n_jobs):
    """
    Applies func to n_jobs slices of args. E.g., if args is an array of 99 items and n_jobs=2, then
    func is applied in two parallel processes to args[0:50] and to args[50:99]

    :param func: function to be parallelized
    :param args: array-like of arguments to be passed to the function in different parallel calls
    :param n_jobs: the number of workers
    """
    args = np.asarray(args)
    slices = _get_parallel_slices(len(args), n_jobs)
    results = Parallel(n_jobs=n_jobs)(
        delayed(func)(args[slice_i]) for slice_i in slices
    )
    return list(itertools.chain.from_iterable(results))


def parallel(func, args, n_jobs):
    """
    A wrapper of multiprocessing:

    >>> Parallel(n_jobs=n_jobs)(
    >>>      delayed(func)(args_i) for args_i in args
    >>> )

    that takes the `quapy.environ` variable as input silently
    """
    def func_dec(environ, *args):
        qp.environ = environ
        return func(*args)
    return Parallel(n_jobs=n_jobs)(
        delayed(func_dec)(qp.environ, args_i) for args_i in args
    )


@contextlib.contextmanager
def temp_seed(seed):
    """
    Can be used in a "with" context to set a temporal seed without modifying the outer numpy's current state. E.g.:

    >>> with temp_seed(random_seed):
    >>>  pass # do any computation depending on np.random functionality

    :param seed: the seed to set within the "with" context
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def download_file(url, archive_filename):
    """
    Downloads a file from a url

    :param url: the url
    :param archive_filename: destination filename
    """
    def progress(blocknum, bs, size):
        total_sz_mb = '%.2f MB' % (size / 1e6)
        current_sz_mb = '%.2f MB' % ((blocknum * bs) / 1e6)
        print('\rdownloaded %s / %s' % (current_sz_mb, total_sz_mb), end='')
    print("Downloading %s" % url)
    urllib.request.urlretrieve(url, filename=archive_filename, reporthook=progress)
    print("")


def download_file_if_not_exists(url, archive_filename):
    """
    Dowloads a function (using :meth:`download_file`) if the file does not exist.

    :param url: the url
    :param archive_filename: destination filename
    """
    if os.path.exists(archive_filename):
        return
    create_if_not_exist(os.path.dirname(archive_filename))
    download_file(url, archive_filename)


def create_if_not_exist(path):
    """
    An alias to `os.makedirs(path, exist_ok=True)` that also returns the path. This is useful in cases like, e.g.:

    >>> path = create_if_not_exist(os.path.join(dir, subdir, anotherdir))

    :param path: path to create
    :return: the path itself
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_quapy_home():
    """
    Gets the home directory of QuaPy, i.e., the directory where QuaPy saves permanent data, such as dowloaded datasets.

    :return: a string representing the path
    """
    home = os.path.join(str(Path.home()), 'quapy_data')
    os.makedirs(home, exist_ok=True)
    return home


def create_parent_dir(path):
    """
    Creates the parent dir (if any) of a given path, if not exists. E.g., for `./path/to/file.txt`, the path `./path/to`
    is created.

    :param path: the path
    """
    parentdir = Path(path).parent
    if parentdir:
        os.makedirs(parentdir, exist_ok=True)


def save_text_file(path, text):
    """
    Saves a text file to disk, given its full path, and creates the parent directory if missing.

    :param path: path where to save the path.
    :param text: text to save.
    """
    create_parent_dir(path)
    with open(text, 'wt') as fout:
        fout.write(text)


def pickled_resource(pickle_path:str, generation_func:callable, *args):
    """
    Allows for fast reuse of resources that are generated only once by calling generation_func(*args). The next times
    this function is invoked, it loads the pickled resource. Example:

    >>> def some_array(n):  # a mock resource created with one parameter (`n`)
    >>>     return np.random.rand(n)
    >>> pickled_resource('./my_array.pkl', some_array, 10)  # the resource does not exist: it is created by calling some_array(10)
    >>> pickled_resource('./my_array.pkl', some_array, 10)  # the resource exists; it is loaded from './my_array.pkl'

    :param pickle_path: the path where to save (first time) and load (next times) the resource
    :param generation_func: the function that generates the resource, in case it does not exist in pickle_path
    :param args: any arg that generation_func uses for generating the resources
    :return: the resource
    """
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


def _check_sample_size(sample_size):
    if sample_size is None:
        assert qp.environ['SAMPLE_SIZE'] is not None, \
            'error: sample_size set to None, and cannot be resolved from the environment'
        sample_size = qp.environ['SAMPLE_SIZE']
    assert isinstance(sample_size, int) and sample_size > 0, \
        'error: sample_size is not a positive integer'
    return sample_size


class EarlyStop:
    """
    A class implementing the early-stopping condition typically used for training neural networks.

    :param patience: the number of (consecutive) times that a monitored evaluation metric (typically obtaind in a
    held-out validation split) can be found to be worse than the best one obtained so far, before flagging the
    stopping condition. An instance of this class is `callable`, and is to be used as follows:

    >>> earlystop = EarlyStop(patience=2, lower_is_better=True)
    >>> earlystop(0.9, epoch=0)
    >>> earlystop(0.7, epoch=1)
    >>> earlystop.IMPROVED  # is True
    >>> earlystop(1.0, epoch=2)
    >>> earlystop.STOP  # is False (patience=1)
    >>> earlystop(1.0, epoch=3)
    >>> earlystop.STOP  # is True (patience=0)
    >>> earlystop.best_epoch  # is 1
    >>> earlystop.best_score  # is 0.7


    :param lower_is_better: if True (default) the metric is to be minimized.

    :ivar best_score: keeps track of the best value seen so far
    :ivar best_epoch: keeps track of the epoch in which the best score was set
    :ivar STOP: flag (boolean) indicating the stopping condition
    :ivar IMPROVED: flag (boolean) indicating whether there was an improvement in the last call

    """

    def __init__(self, patience, lower_is_better=True):

        self.PATIENCE_LIMIT = patience
        self.better = lambda a,b: a<b if lower_is_better else a>b
        self.patience = patience
        self.best_score = None
        self.best_epoch = None
        self.STOP = False
        self.IMPROVED = False

    def __call__(self, watch_score, epoch):
        """
        Commits the new score found in epoch `epoch`. If the score improves over the best score found so far, then
        the patiente counter gets reset. If otherwise, the patience counter is decreased, and in case it reachs 0,
        the flag STOP becomes True.

        :param watch_score: the new score
        :param epoch: the current epoch
        """
        self.IMPROVED = (self.best_score is None or self.better(watch_score, self.best_score))
        if self.IMPROVED:
            self.best_score = watch_score
            self.best_epoch = epoch
            self.patience = self.PATIENCE_LIMIT
        else:
            self.patience -= 1
            if self.patience <= 0:
                self.STOP = True