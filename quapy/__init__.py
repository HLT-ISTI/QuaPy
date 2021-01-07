from . import error
from .data import datasets
from . import functional
from . import method
from . import data
from . import evaluation
from . import plot
from . import util
from method.aggregative import isaggregative, isprobabilistic


environ = {
    'SAMPLE_SIZE': None,
    'UNK_TOKEN': '[UNK]',
    'UNK_INDEX': 0,
    'PAD_TOKEN': '[PAD]',
    'PAD_INDEX': 1,
}


def isbinary(x):
    return data.isbinary(x) or method.isbinary(x)