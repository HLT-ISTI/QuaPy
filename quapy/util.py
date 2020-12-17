import itertools
import multiprocessing
from joblib import Parallel, delayed
import contextlib
import numpy as np
import urllib
import os
from pathlib import Path


def get_parallel_slices(n_tasks, n_jobs=-1):
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    batch = int(n_tasks / n_jobs)
    remainder = n_tasks % n_jobs
    return [slice(job * batch, (job + 1) * batch + (remainder if job == n_jobs - 1 else 0)) for job in
            range(n_jobs)]


def parallelize(func, args, n_jobs):
    args = np.asarray(args)
    slices = get_parallel_slices(len(args), n_jobs)
    results = Parallel(n_jobs=n_jobs)(
        delayed(func)(args[slice_i]) for slice_i in slices
    )
    return list(itertools.chain.from_iterable(results))


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
    return os.path.join(str(Path.home()), 'quapy_data')