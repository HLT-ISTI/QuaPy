import itertools
import signal
from copy import deepcopy
from typing import Union, Callable

import numpy as np

import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.evaluation import artificial_prevalence_prediction, natural_prevalence_prediction, gen_prevalence_prediction
from quapy.method.aggregative import BaseQuantifier
import inspect

from util import _check_sample_size


class GridSearchQ(BaseQuantifier):
    """Grid Search optimization targeting a quantification-oriented metric.

    Optimizes the hyperparameters of a quantification method, based on an evaluation method and on an evaluation
    protocol for quantification.

    :param model: the quantifier to optimize
    :type model: BaseQuantifier
    :param param_grid: a dictionary with keys the parameter names and values the list of values to explore
    :param sample_size: the size of the samples to extract from the validation set (ignored if protocl='gen')
    :param protocol: either 'app' for the artificial prevalence protocol, 'npp' for the natural prevalence
        protocol, or 'gen' for using a custom sampling generator function
    :param n_prevpoints: if specified, indicates the number of equally distant points to extract from the interval
        [0,1] in order to define the prevalences of the samples; e.g., if n_prevpoints=5, then the prevalences for
        each class will be explored in [0.00, 0.25, 0.50, 0.75, 1.00]. If not specified, then eval_budget is requested.
        Ignored if protocol!='app'.
    :param n_repetitions: the number of repetitions for each combination of prevalences. This parameter is ignored
        for the protocol='app' if eval_budget is set and is lower than the number of combinations that would be
        generated using the value assigned to n_prevpoints (for the current number of classes and n_repetitions).
        Ignored for protocol='npp' and protocol='gen' (use eval_budget for setting a maximum number of samples in
        those cases).
    :param eval_budget: if specified, sets a ceil on the number of evaluations to perform for each hyper-parameter
        combination. For example, if protocol='app', there are 3 classes, n_repetitions=1 and eval_budget=20, then
        n_prevpoints will be set to 5, since this will generate 15 different prevalences, i.e., [0, 0, 1],
        [0, 0.25, 0.75], [0, 0.5, 0.5] ... [1, 0, 0], and since setting it to 6 would generate more than
        20. When protocol='gen', indicates the maximum number of samples to generate, but less samples will be
        generated if the generator yields less samples.
    :param error: an error function (callable) or a string indicating the name of an error function (valid ones
        are those in qp.error.QUANTIFICATION_ERROR
    :param refit: whether or not to refit the model on the whole labelled collection (training+validation) with
        the best chosen hyperparameter combination. Ignored if protocol='gen'
    :param val_split: either a LabelledCollection on which to test the performance of the different settings, or
        a float in [0,1] indicating the proportion of labelled data to extract from the training set, or a callable
        returning a generator function each time it is invoked (only for protocol='gen').
    :param n_jobs: number of parallel jobs
    :param random_seed: set the seed of the random generator to replicate experiments. Ignored if protocol='gen'.
    :param timeout: establishes a timer (in seconds) for each of the hyperparameters configurations being tested.
        Whenever a run takes longer than this timer, that configuration will be ignored. If all configurations end up
        being ignored, a TimeoutError exception is raised. If -1 (default) then no time bound is set.
    :param verbose: set to True to get information through the stdout
    """

    def __init__(self,
                 model: BaseQuantifier,
                 param_grid: dict,
                 sample_size: Union[int, None] = None,
                 protocol='app',
                 n_prevpoints: int = None,
                 n_repetitions: int = 1,
                 eval_budget: int = None,
                 error: Union[Callable, str] = qp.error.mae,
                 refit=True,
                 val_split=0.4,
                 n_jobs=1,
                 random_seed=42,
                 timeout=-1,
                 verbose=False):

        self.model = model
        self.param_grid = param_grid
        self.sample_size = sample_size
        self.protocol = protocol.lower()
        self.n_prevpoints = n_prevpoints
        self.n_repetitions = n_repetitions
        self.eval_budget = eval_budget
        self.refit = refit
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.timeout = timeout
        self.verbose = verbose
        self.__check_error(error)
        assert self.protocol in {'app', 'npp', 'gen'}, \
            'unknown protocol: valid ones are "app" or "npp" for the "artificial" or the "natural" prevalence ' \
            'protocols. Use protocol="gen" when passing a generator function thorough val_split that yields a ' \
            'sample (instances) and their prevalence (ndarray) at each iteration.'
        assert self.eval_budget is None or isinstance(self.eval_budget, int)
        if self.protocol in ['npp', 'gen']:
            if self.protocol=='npp' and (self.eval_budget is None or self.eval_budget <= 0):
                raise ValueError(f'when protocol="npp" the parameter eval_budget should be '
                                 f'indicated (and should be >0).')
            if self.n_repetitions != 1:
                print('[warning] n_repetitions has been set and will be ignored for the selected protocol')

    def _sout(self, msg):
        if self.verbose:
            print(f'[{self.__class__.__name__}]: {msg}')

    def __check_training_validation(self, training, validation):
        if isinstance(validation, LabelledCollection):
            return training, validation
        elif isinstance(validation, float):
            assert 0. < validation < 1., 'validation proportion should be in (0,1)'
            training, validation = training.split_stratified(train_prop=1 - validation, random_state=self.random_seed)
            return training, validation
        elif self.protocol=='gen' and inspect.isgenerator(validation()):
            return training, validation
        else:
            raise ValueError(f'"validation" must either be a LabelledCollection or a float in (0,1) indicating the'
                             f'proportion of training documents to extract (type found: {type(validation)}). '
                             f'Optionally, "validation" can be a callable function returning a generator that yields '
                             f'the sample instances along with their true prevalence at each iteration by '
                             f'setting protocol="gen".')

    def __check_error(self, error):
        if error in qp.error.QUANTIFICATION_ERROR:
            self.error = error
        elif isinstance(error, str):
            self.error = qp.error.from_name(error)
        elif hasattr(error, '__call__'):
            self.error = error
        else:
            raise ValueError(f'unexpected error type; must either be a callable function or a str representing\n'
                             f'the name of an error function in {qp.error.QUANTIFICATION_ERROR_NAMES}')

    def __generate_predictions(self, model, val_split):
        commons = {
            'n_repetitions': self.n_repetitions,
            'n_jobs': self.n_jobs,
            'random_seed': self.random_seed,
            'verbose': False
        }
        if self.protocol == 'app':
            return artificial_prevalence_prediction(
                model, val_split, self.sample_size,
                n_prevpoints=self.n_prevpoints,
                eval_budget=self.eval_budget,
                **commons
            )
        elif self.protocol == 'npp':
            return natural_prevalence_prediction(
                model, val_split, self.sample_size,
                **commons)
        elif self.protocol == 'gen':
            return gen_prevalence_prediction(model, gen_fn=val_split, eval_budget=self.eval_budget)
        else:
            raise ValueError('unknown protocol')

    def fit(self, training: LabelledCollection, val_split: Union[LabelledCollection, float, Callable] = None):
        """ Learning routine. Fits methods with all combinations of hyperparameters and selects the one minimizing
            the error metric.

        :param training: the training set on which to optimize the hyperparameters
        :param val_split: either a LabelledCollection on which to test the performance of the different settings, or
            a float in [0,1] indicating the proportion of labelled data to extract from the training set
        :return: self
        """
        if val_split is None:
            val_split = self.val_split
        training, val_split = self.__check_training_validation(training, val_split)
        if self.protocol != 'gen':
            self.sample_size = _check_sample_size(self.sample_size)

        params_keys = list(self.param_grid.keys())
        params_values = list(self.param_grid.values())

        model = self.model

        if self.timeout > 0:
            def handler(signum, frame):
                self._sout('timeout reached')
                raise TimeoutError()

            signal.signal(signal.SIGALRM, handler)

        self.param_scores_ = {}
        self.best_score_ = None
        some_timeouts = False
        for values in itertools.product(*params_values):
            params = dict({k: values[i] for i, k in enumerate(params_keys)})

            if self.timeout > 0:
                signal.alarm(self.timeout)

            try:
                # overrides default parameters with the parameters being explored at this iteration
                model.set_params(**params)
                model.fit(training)
                true_prevalences, estim_prevalences = self.__generate_predictions(model, val_split)
                score = self.error(true_prevalences, estim_prevalences)

                self._sout(f'checking hyperparams={params} got {self.error.__name__} score {score:.5f}')
                if self.best_score_ is None or score < self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = params
                    self.best_model_ = deepcopy(model)
                self.param_scores_[str(params)] = score

                if self.timeout > 0:
                    signal.alarm(0)
            except TimeoutError:
                print(f'timeout reached for config {params}')
                some_timeouts = True

        if self.best_score_ is None and some_timeouts:
            raise TimeoutError('all jobs took more than the timeout time to end')

        self._sout(f'optimization finished: best params {self.best_params_} (score={self.best_score_:.5f})')

        if self.refit:
            self._sout(f'refitting on the whole development set')
            self.best_model_.fit(training + val_split)

        return self

    def quantify(self, instances):
        """Estimate class prevalence values using the best model found after calling the :meth:`fit` method.

        :param instances: sample contanining the instances
        :return: a ndarray of shape `(n_classes)` with class prevalence estimates as according to the best model found
            by the model selection process.
        """
        assert hasattr(self, 'best_model_'), 'quantify called before fit'
        return self.best_model().quantify(instances)

    @property
    def classes_(self):
        """
        Classes on which the quantifier has been trained on.
        :return: a ndarray of shape `(n_classes)` with the class identifiers
        """
        return self.best_model().classes_

    def set_params(self, **parameters):
        """Sets the hyper-parameters to explore.

        :param parameters: a dictionary with keys the parameter names and values the list of values to explore
        """
        self.param_grid = parameters

    def get_params(self, deep=True):
        """Returns the dictionary of hyper-parameters to explore (`param_grid`)

        :param deep: Unused
        :return: the dictionary `param_grid`
        """
        return self.param_grid

    def best_model(self):
        """
        Returns the best model found after calling the :meth:`fit` method, i.e., the one trained on the combination
        of hyper-parameters that minimized the error function.

        :return: a trained quantifier
        """
        if hasattr(self, 'best_model_'):
            return self.best_model_
        raise ValueError('best_model called before fit')
