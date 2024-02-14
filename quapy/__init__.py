"""QuaPy module for quantification"""
from quapy.data import datasets
from . import error
from . import data
from . import functional
# from . import method
from . import evaluation
from . import protocol
from . import plot
from . import util
from . import model_selection
from . import classification

__version__ = '0.1.8'

environ = {
    'SAMPLE_SIZE': None,
    'UNK_TOKEN': '[UNK]',
    'UNK_INDEX': 0,
    'PAD_TOKEN': '[PAD]',
    'PAD_INDEX': 1,
    'SVMPERF_HOME': './svm_perf_quantification',
    'N_JOBS': 1
}


def _get_njobs(n_jobs):
    """
    If `n_jobs` is None, then it returns `environ['N_JOBS']`;
    if otherwise, returns `n_jobs`.

    :param n_jobs: the number of `n_jobs` or None if not specified
    :return: int
    """
    return environ['N_JOBS'] if n_jobs is None else n_jobs


def _get_sample_size(sample_size):
    """
    If `sample_size` is None, then it returns `environ['SAMPLE_SIZE']`;
    if otherwise, returns `sample_size`.
    If none of these are set, then a ValueError exception is raised.

    :param sample_size: integer or None
    :return: int
    """
    sample_size = environ['SAMPLE_SIZE'] if sample_size is None else sample_size
    if sample_size is None:
        raise ValueError('neither sample_size nor qp.environ["SAMPLE_SIZE"] have been specified')
    return sample_size
