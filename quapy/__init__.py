from . import error
from . import data
from quapy.data import datasets
from . import functional
from . import method
from . import evaluation
from . import plot
from . import util
from . import model_selection
from . import classification
from quapy.method.base import isprobabilistic, isaggregative

__version__ = '0.1.6'

environ = {
    'SAMPLE_SIZE': None,
    'UNK_TOKEN': '[UNK]',
    'UNK_INDEX': 0,
    'PAD_TOKEN': '[PAD]',
    'PAD_INDEX': 1,
    'SVMPERF_HOME': './svm_perf_quantification'
}

def isbinary(x):
    return x.binary