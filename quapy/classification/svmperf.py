import random
import shutil
import subprocess
import tempfile
from os import remove, makedirs
from os.path import join, exists
from subprocess import PIPE, STDOUT
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import dump_svmlight_file


class SVMperf(BaseEstimator, ClassifierMixin):
    """A wrapper for the `SVM-perf package <https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html>`__ by Thorsten Joachims.
    When using losses for quantification, the source code has to be patched. See
    the `installation documentation <https://hlt-isti.github.io/QuaPy/build/html/Installation.html#svm-perf-with-quantification-oriented-losses>`__
    for further details.

    References:

        * `Esuli et al.2015 <https://dl.acm.org/doi/abs/10.1145/2700406?casa_token=8D2fHsGCVn0AAAAA:ZfThYOvrzWxMGfZYlQW_y8Cagg-o_l6X_PcF09mdETQ4Tu7jK98mxFbGSXp9ZSO14JkUIYuDGFG0>`__
        * `Barranquero et al.2015 <https://www.sciencedirect.com/science/article/abs/pii/S003132031400291X>`__

    :param svmperf_base: path to directory containing the binary files `svm_perf_learn` and `svm_perf_classify`
    :param C: trade-off between training error and margin (default 0.01)
    :param verbose: set to True to print svm-perf std outputs
    :param loss: the loss to optimize for. Available losses are "01", "f1", "kld", "nkld", "q", "qacc", "qf1", "qgm", "mae", "mrae".
    :param host_folder: directory where to store the trained model; set to None (default) for using a tmp directory
        (temporal directories are automatically deleted)
    """

    # losses with their respective codes in svm_perf implementation
    valid_losses = {'01':0, 'f1':1, 'kld':12, 'nkld':13, 'q':22, 'qacc':23, 'qf1':24, 'qgm':25, 'mae':26, 'mrae':27}

    def __init__(self, svmperf_base, C=0.01, verbose=False, loss='01', host_folder=None):
        assert exists(svmperf_base), f'path {svmperf_base} does not seem to point to a valid path'
        self.svmperf_base = svmperf_base
        self.C = C
        self.verbose = verbose
        self.loss = loss
        self.host_folder = host_folder

    # def set_params(self, **parameters):
    #     """
    #     Set the hyper-parameters for svm-perf. Currently, only the `C` and `loss` parameters are supported
    #
    #     :param parameters: a `**kwargs` dictionary `{'C': <float>}`
    #     """
    #     assert sorted(list(parameters.keys())) == ['C', 'loss'], \
    #         'currently, only the C and loss parameters are supported'
    #     self.C = parameters.get('C', self.C)
    #     self.loss = parameters.get('loss', self.loss)
    #
    # def get_params(self, deep=True):
    #     return {'C': self.C, 'loss': self.loss}

    def fit(self, X, y):
        """
        Trains the SVM for the multivariate performance loss

        :param X: training instances
        :param y: a binary vector of labels
        :return: `self`
        """
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
        random_code = 'svmperfprocess'+'-'.join(str(local_random.randint(0, 1000000)) for _ in range(5))
        if self.host_folder is None:
            # tmp dir are removed after the fit terminates in multiprocessing...
            self.tmpdir = tempfile.TemporaryDirectory(suffix=random_code).name
        else:
            self.tmpdir = join(self.host_folder, '.' + random_code)
        makedirs(self.tmpdir, exist_ok=True)

        self.model = join(self.tmpdir, 'model-'+random_code)
        traindat = join(self.tmpdir, f'train-{random_code}.dat')

        dump_svmlight_file(X, y, traindat, zero_based=False)

        cmd = ' '.join([self.svmperf_learn, self.c_cmd, self.loss_cmd, traindat, self.model])
        if self.verbose:
            print('[Running]', cmd)
        p = subprocess.run(cmd.split(), stdout=PIPE, stderr=STDOUT)
        if not exists(self.model):
            print(p.stderr.decode('utf-8'))
        remove(traindat)

        if self.verbose:
            print(p.stdout.decode('utf-8'))

        return self

    def predict(self, X):
        """
        Predicts labels for the instances `X`

        :param X: array-like of shape `(n_samples, n_features)` instances to classify
        :return: a `numpy` array of length `n` containing the label predictions, where `n` is the number of
            instances in `X`
        """
        confidence_scores = self.decision_function(X)
        predictions = (confidence_scores > 0) * 1
        return predictions

    def decision_function(self, X, y=None):
        """
        Evaluate the decision function for the samples in `X`.

        :param X: array-like of shape `(n_samples, n_features)` containing the instances to classify
        :param y: unused
        :return: array-like of shape `(n_samples,)` containing the decision scores of the instances
        """
        assert hasattr(self, 'tmpdir'), 'predict called before fit'
        assert self.tmpdir is not None, 'model directory corrupted'
        assert exists(self.model), 'model not found'
        if y is None:
            y = np.zeros(X.shape[0])

        # in order to allow for parallel runs of predict, a random code is assigned
        local_random = random.Random()
        random_code = '-'.join(str(local_random.randint(0, 1000000)) for _ in range(5))
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
            shutil.rmtree(self.tmpdir, ignore_errors=True)

