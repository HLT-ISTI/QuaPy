import random
import subprocess
import tempfile
from os.path import join, exists
from os import remove
from subprocess import PIPE, STDOUT
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import dump_svmlight_file


class SVMperf(BaseEstimator, ClassifierMixin):

    # losses with their respective codes in svm_perf implementation
    valid_losses = {'01':0, 'f1':1, 'kld':12, 'nkld':13, 'q':22, 'qacc':23, 'qf1':24, 'qgm':25, 'mae':26, 'mrae':27}

    def __init__(self, svmperf_base, C=0.01, verbose=False, loss='01'):
        self.svmperf_base = svmperf_base
        self.C = C
        self.verbose = verbose
        self.loss = loss

    def set_c(self, C):
        self.param_C = '-c ' + str(C)

    def set_params(self, **parameters):
        assert list(parameters.keys()) == ['C'], 'currently, only the C parameter is supported'
        self.set_c(parameters['C'])

    def fit(self, X, y):
        assert self.loss in SVMperf.valid_losses, \
            f'unsupported loss {self.loss}, valid ones are {list(SVMperf.valid_losses.keys())}'

        self.svmperf_learn = join(self.svmperf_base, 'svm_perf_learn')
        self.svmperf_classify = join(self.svmperf_base, 'svm_perf_classify')
        self.loss_cmd = '-l ' + str(self.valid_losses[self.loss])
        self.set_c(self.C)

        self.classes_ = sorted(np.unique(y))
        self.n_classes_ = len(self.classes_)

        local_random = random.Random()
        # this would allow to run parallel instances of predict
        random_code = '-'.join(str(local_random.randint(0,1000000)) for _ in range(5))
        self.tmpdir = tempfile.TemporaryDirectory(suffix=random_code)

        self.model = join(self.tmpdir.name, 'model-'+random_code)
        traindat = join(self.tmpdir.name, f'train-{random_code}.dat')

        dump_svmlight_file(X, y, traindat, zero_based=False)

        cmd = ' '.join([self.svmperf_learn, self.param_C, self.loss_cmd, traindat, self.model])
        if self.verbose:
            print('[Running]', cmd)
        p = subprocess.run(cmd.split(), stdout=PIPE, stderr=STDOUT)
        remove(traindat)

        if self.verbose:
            print(p.stdout.decode('utf-8'))

        return self

    def predict(self, X, y=None):
        confidence_scores = self.decision_function(X)
        predictions = (confidence_scores > 0) * 1
        return predictions

    def decision_function(self, X, y=None):
        assert hasattr(self, 'tmpdir'), 'predict called before fit'
        assert self.tmpdir is not None, 'model directory corrupted'
        assert exists(self.model), 'model not found'
        if y is None:
            y = np.zeros(X.shape[0])

        # in order to allow for parallel runs of predict, a random code is assigned
        local_random = random.Random()
        random_code = '-'.join(str(local_random.randint(0, 1000000)) for _ in range(5))
        predictions_path = join(self.tmpdir.name, 'predictions'+random_code+'.dat')
        testdat = join(self.tmpdir.name, 'test'+random_code+'.dat')
        dump_svmlight_file(X, y, testdat, zero_based=False)

        cmd = ' '.join([self.svmperf_classify, testdat, self.model, predictions_path])
        if self.verbose:
            print('[Running]', cmd)
        p = subprocess.run(cmd.split(), stdout=PIPE, stderr=STDOUT)

        if self.verbose:
            print(p.stdout.decode('utf-8'))

        scores = np.loadtxt(predictions_path)
        remove(testdat)
        remove(predictions_path)

        return scores


