"""QuaPy module for quantification"""
from sklearn.linear_model import LogisticRegression

from quapy.data import datasets
from . import error
from . import data
from . import functional
from . import method
from . import evaluation
from . import protocol
from . import plot
from . import util
from . import model_selection
from . import classification
import os

__version__ = '0.1.9'

environ = {
    'SAMPLE_SIZE': None,
    'UNK_TOKEN': '[UNK]',
    'UNK_INDEX': 0,
    'PAD_TOKEN': '[PAD]',
    'PAD_INDEX': 1,
    'SVMPERF_HOME': './svm_perf_quantification',
    'N_JOBS': int(os.getenv('N_JOBS', 1)),
    'DEFAULT_CLS': LogisticRegression(max_iter=3000)
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


def _get_classifier(classifier):
    """
    If `classifier` is None, then it returns `environ['DEFAULT_CLS']`;
    if otherwise, returns `classifier`.

    :param classifier: sklearn's estimator or None
    :return: sklearn's estimator
    """
    if classifier is None:
        from sklearn.base import clone
        classifier = clone(environ['DEFAULT_CLS'])
    if classifier is None:
        raise ValueError('neither classifier nor qp.environ["DEFAULT_CLS"] have been specified')
    return classifier
