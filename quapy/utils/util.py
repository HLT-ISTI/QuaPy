import itertools
import multiprocessing
from joblib import Parallel, delayed
import contextlib
import numpy as np


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


