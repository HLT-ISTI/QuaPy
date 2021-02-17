import random
import subprocess
import tempfile
from os import remove, makedirs
from os.path import join, exists
from subprocess import PIPE, STDOUT
import shutil

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import dump_svmlight_file


class SVMperf(BaseEstimator, ClassifierMixin):

    # losses with their respective codes in svm_perf implementation
    valid_losses = {'01':0, 'f1':1, 'kld':12, 'nkld':13, 'q':22, 'qacc':23, 'qf1':24, 'qgm':25, 'mae':26, 'mrae':27}

    def __init__(self, svmperf_base, C=0.01, verbose=False, loss='01'):
        assert exists(svmperf_base), f'path {svmperf_base} does not seem to point to a valid path'
        self.svmperf_base = svmperf_base
        self.C = C
        self.verbose = verbose
        self.loss = loss

    def set_params(self, **parameters):
        assert list(parameters.keys()) == ['C'], 'currently, only the C parameter is supported'
        self.C = parameters['C']

    def fit(self, X, y):
        assert self.loss in SVMperf.valid_losses, \
            f'unsupported loss {self.loss}, valid ones are {list(SVMperf.valid_losses.keys())}'

        self.svmperf_learn = join(self.svmperf_base, 'svm_perf_learn')
        self.svmperf_classify = join(self.svmperf_base, 'svm_perf_classify')
        self.loss_cmd = '-w 3 -l ' + str(self.valid_losses[self.loss])
        self.c_cmd = '-c ' + str(self.C)

        self.classes_ = sorted(np.unique(y))
        self.n_classes_ = len(self.classes_)

        local_random = random.Random()
        # this would allow to run parallel instances of predict
        random_code = '-'.join(str(local_random.randint(0,1000000)) for _ in range(5))
        # self.tmpdir = tempfile.TemporaryDirectory(suffix=random_code)
        # tmp dir are removed after the fit terminates in multiprocessing... moving to regular directories + __del__
        self.tmpdir = '.svmperf-' + random_code
        makedirs(self.tmpdir, exist_ok=True)

        # self.model = join(self.tmpdir.name, 'model-'+random_code)
        # traindat = join(self.tmpdir.name, f'train-{random_code}.dat')
        self.model = join(self.tmpdir, 'model-'+random_code)
        traindat = join(self.tmpdir, f'train-{random_code}.dat')

        dump_svmlight_file(X, y, traindat, zero_based=False)

        cmd = ' '.join([self.svmperf_learn, self.c_cmd, self.loss_cmd, traindat, self.model])
        if self.verbose:
            print('[Running]', cmd)
        p = subprocess.run(cmd.split(), stdout=PIPE, stderr=STDOUT)
        remove(traindat)

        if self.verbose:
            print(p.stdout.decode('utf-8'))

        return self

    def predict(self, X):
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
        # predictions_path = join(self.tmpdir.name, 'predictions'+random_code+'.dat')
        # testdat = join(self.tmpdir.name, 'test'+random_code+'.dat')
        predictions_path = join(self.tmpdir, 'predictions' + random_code + '.dat')
        testdat = join(self.tmpdir, 'test' + random_code + '.dat')
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

    def __del__(self):
        if hasattr(self, 'tmpdir'):
            shutil.rmtree(self.tmpdir)

