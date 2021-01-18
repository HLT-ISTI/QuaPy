from sklearn.linear_model import LogisticRegression
import quapy as qp
from classification.methods import PCALR
from method.meta import QuaNet
from quapy.method.aggregative import *
from NewMethods.methods import *
from experiments import run, SAMPLE_SIZE
import numpy as np
import itertools
from joblib import Parallel, delayed
import settings
import argparse
import torch

parser = argparse.ArgumentParser(description='Run experiments for Tweeter Sentiment Quantification')
parser.add_argument('results', metavar='RESULT_PATH', type=str, help='path to the directory where to store the results')
parser.add_argument('svmperfpath', metavar='SVMPERF_PATH', type=str, help='path to the directory with svmperf')
args = parser.parse_args()


def quantification_models():
    def newLR():
        return LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
    __C_range = np.logspace(-4, 5, 10)
    lr_params = {'C': __C_range, 'class_weight': [None, 'balanced']}
    svmperf_params = {'C': __C_range}
    #yield 'paccsld', PACCSLD(newLR()), lr_params
    #yield 'hdysld', OneVsAll(HDySLD(newLR())), lr_params  # <-- promising!

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running QuaNet in {device}')
    yield 'quanet', QuaNet(PCALR(**newLR().get_params()), SAMPLE_SIZE, device=device), lr_params


if __name__ == '__main__':

    print(f'Result folder: {args.results}')
    np.random.seed(0)

    optim_losses = ['mae']
    datasets = ['hcr']  # qp.datasets.TWITTER_SENTIMENT_DATASETS_TRAIN
    models = quantification_models()

    results = Parallel(n_jobs=settings.N_JOBS)(
        delayed(run)(experiment) for experiment in itertools.product(optim_losses, datasets, models)
    )


