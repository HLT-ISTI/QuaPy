import itertools
import signal
from copy import deepcopy
from enum import Enum
from typing import Union, Callable
from functools import wraps

import numpy as np
from sklearn import clone

import quapy as qp
from quapy import evaluation
from quapy.protocol import AbstractProtocol, OnLabelledCollectionProtocol
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import BaseQuantifier, AggregativeQuantifier
from quapy.util import timeout
from time import time


class Status(Enum):
    SUCCESS = 1
    TIMEOUT = 2
    INVALID = 3
    ERROR = 4

def check_status(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        obj = args[0]
        tinit = time()

        job_descriptor = dict(args[1])
        params = {**job_descriptor.get('cls-params', {}), **job_descriptor.get('q-params', {})}

        if obj.timeout > 0:
            def handler(signum, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(obj.timeout)

        try:
            job_descriptor = func(*args, **kwargs)

            ttime = time() - tinit

            score = job_descriptor.get('score', None)
            if score is not None:
                obj._sout(f'hyperparams=[{params}]\t got {obj.error.__name__} = {score:.5f} [took {ttime:.4f}s]')

            if obj.timeout > 0:
                signal.alarm(0)

            exit_status = Status.SUCCESS

        except TimeoutError:
            obj._sout(f'timeout ({obj.timeout}s) reached for config {params}')
            exit_status = Status.TIMEOUT

        except ValueError as e:
            obj._sout(f'the combination of hyperparameters {params} is invalid')
            obj._sout(f'\tException: {e}')
            exit_status = Status.INVALID

        except Exception as e:
            obj._sout(f'something went wrong for config {params}; skipping:')
            obj._sout(f'\tException: {e}')
            exit_status = Status.ERROR

        job_descriptor['status'] = exit_status
        job_descriptor['params'] = params
        return job_descriptor
    return wrapper


class GridSearchQ(BaseQuantifier):
    """Grid Search optimization targeting a quantification-oriented metric.

    Optimizes the hyperparameters of a quantification method, based on an evaluation method and on an evaluation
    protocol for quantification.

    :param model: the quantifier to optimize
    :type model: BaseQuantifier
    :param param_grid: a dictionary with keys the parameter names and values the list of values to explore
    :param protocol: a sample generation protocol, an instance of :class:`quapy.protocol.AbstractProtocol`
    :param error: an error function (callable) or a string indicating the name of an error function (valid ones
        are those in :class:`quapy.error.QUANTIFICATION_ERROR`
    :param refit: whether or not to refit the model on the whole labelled collection (training+validation) with
        the best chosen hyperparameter combination. Ignored if protocol='gen'
    :param timeout: establishes a timer (in seconds) for each of the hyperparameters configurations being tested.
        Whenever a run takes longer than this timer, that configuration will be ignored. If all configurations end up
        being ignored, a TimeoutError exception is raised. If -1 (default) then no time bound is set.
    :param verbose: set to True to get information through the stdout
    """

    def __init__(self,
                 model: BaseQuantifier,
                 param_grid: dict,
                 protocol: AbstractProtocol,
                 error: Union[Callable, str] = qp.error.mae,
                 refit=True,
                 timeout=-1,
                 n_jobs=None,
                 verbose=False):

        self.model = model
        self.param_grid = param_grid
        self.protocol = protocol
        self.refit = refit
        self.timeout = timeout
        self.n_jobs = qp._get_njobs(n_jobs)
        self.verbose = verbose
        self.__check_error(error)
        assert isinstance(protocol, AbstractProtocol), 'unknown protocol'

    def _sout(self, msg):
        if self.verbose:
            print(f'[{self.__class__.__name__}:{self.model.__class__.__name__}]: {msg}')

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

    def _prepare_classifier(self, args):
        cls_params = args['cls-params']
        training = args['training']
        model = deepcopy(self.model)
        model.set_params(**cls_params)
        predictions = model.classifier_fit_predict(training)
        return {'model': model, 'predictions': predictions, 'cls-params': cls_params}

    def _prepare_aggregation(self, args):

        model = args['model']
        predictions = args['predictions']
        cls_params = args['cls-params']
        q_params = args['q-params']
        training = args['training']

        params = {**cls_params, **q_params}

        def job(model):
            tinit = time()
            model = deepcopy(model)
            # overrides default parameters with the parameters being explored at this iteration
            model.set_params(**q_params)
            model.aggregation_fit(predictions, training)
            score = evaluation.evaluate(model, protocol=self.protocol, error_metric=self.error)
            ttime = time()-tinit

            return {
                'model': model,
                'cls-params':cls_params,
                'q-params': q_params,
                'params': params,
                'score': score,
                'ttime':ttime
            }

        out, status = self._error_handler(job, args)
        if status == Status.SUCCESS:
            self._sout(f'hyperparams=[{params}]\t got {self.error.__name__} = {out["score"]:.5f} [took {out["time"]:.4f}s]')
        elif status == Status.INVALID:
            self._sout(f'the combination of hyperparameters {params} is invalid')
        elif status == Status.


    def _prepare_model(self, args):
        params, training = args
        model = deepcopy(self.model)
        # overrides default parameters with the parameters being explored at this iteration
        model.set_params(**params)
        model.fit(training)
        score = evaluation.evaluate(model, protocol=self.protocol, error_metric=self.error)
        return {'model': model, 'params': params, 'score': score}


    def _compute_scores_aggregative(self, training):

        # break down the set of hyperparameters into two: classifier-specific, quantifier-specific
        cls_configs, q_configs = group_params(self.param_grid)

        # train all classifiers and get the predictions
        partial_setups = qp.util.parallel(
            self._prepare_classifier,
            ({'cls-params':params, 'training':training} for params in cls_configs),
            seed=qp.environ.get('_R_SEED', None),
            n_jobs=self.n_jobs,
            asarray=False,
        )

        # filter out classifier configurations that yield any error
        for setup in partial_setups:
            if setup['status'] != Status.SUCCESS:
                self._sout(f'-> classifier hyperparemters {setup["params"]} caused '
                           f'error {setup["status"]} and will be ignored')

        partial_setups = [setup for setup in partial_setups if setup['status']==Status.SUCCESS]

        if len(partial_setups) == 0:
            raise ValueError('No valid configuration found for the classifier.')

        # explore the quantifier-specific hyperparameters for each training configuration
        scores = qp.util.parallel(
            self._prepare_aggregation,
            ({'q-params': setup[1], 'training': training, **setup[0]} for setup in itertools.product(partial_setups, q_configs)),
            seed=qp.environ.get('_R_SEED', None),
            n_jobs=self.n_jobs
        )

        return scores

    def _compute_scores_nonaggregative(self, training):
        configs = expand_grid(self.param_grid)

        # pass a seed to parallel, so it is set in child processes
        scores = qp.util.parallel(
            self._prepare_model,
            ((params, training) for params in configs),
            seed=qp.environ.get('_R_SEED', None),
            n_jobs=self.n_jobs
        )
        return scores

    def _compute_scores(self, training):
        if isinstance(self.model, AggregativeQuantifier):
            return self._compute_scores_aggregative(training)
        else:
            return self._compute_scores_nonaggregative(training)

    def fit(self, training: LabelledCollection):
        """ Learning routine. Fits methods with all combinations of hyperparameters and selects the one minimizing
            the error metric.

        :param training: the training set on which to optimize the hyperparameters
        :return: self
        """

        if self.refit and not isinstance(self.protocol, OnLabelledCollectionProtocol):
            raise RuntimeWarning(
                f'"refit" was requested, but the protocol does not implement '
                f'the {OnLabelledCollectionProtocol.__name__} interface'
            )

        tinit = time()

        self._sout(f'starting model selection with n_jobs={self.n_jobs}')
        results = self._compute_scores(training)

        self.param_scores_ = {}
        self.best_score_ = None
        for job_result in results:
            score = job_result.get('score', None)
            params = job_result['params']
            if score is not None:
                if self.best_score_ is None or score < self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = params
                    self.best_model_ = job_result['model']
                self.param_scores_[str(params)] = score
            else:
                self.param_scores_[str(params)] = job_result['status']

        tend = time()-tinit

        if self.best_score_ is None:
            raise TimeoutError('no combination of hyperparameters seem to work')

        self._sout(f'optimization finished: best params {self.best_params_} (score={self.best_score_:.5f}) '
                   f'[took {tend:.4f}s]')

        if self.refit:
            if isinstance(self.protocol, OnLabelledCollectionProtocol):
                tinit = time()
                self._sout(f'refitting on the whole development set')
                self.best_model_.fit(training + self.protocol.get_labelled_collection())
                tend = time() - tinit
                self.refit_time_ = tend
            else:
                raise RuntimeWarning(f'the model cannot be refit on the whole dataset')

        return self

    def quantify(self, instances):
        """Estimate class prevalence values using the best model found after calling the :meth:`fit` method.

        :param instances: sample contanining the instances
        :return: a ndarray of shape `(n_classes)` with class prevalence estimates as according to the best model found
            by the model selection process.
        """
        assert hasattr(self, 'best_model_'), 'quantify called before fit'
        return self.best_model().quantify(instances)

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


    def _error_handler(self, func, *args, **kwargs):

        try:
            with timeout(self.timeout):
                output = func(*args, **kwargs)
                return output, Status.SUCCESS

        except TimeoutError:
            return None, Status.TIMEOUT

        except ValueError:
            return None, Status.INVALID

        except Exception:
            return None, Status.ERROR



def cross_val_predict(quantifier: BaseQuantifier, data: LabelledCollection, nfolds=3, random_state=0):
    """
    Akin to `scikit-learn's cross_val_predict <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html>`_
    but for quantification.

    :param quantifier: a quantifier issuing class prevalence values
    :param data: a labelled collection
    :param nfolds: number of folds for k-fold cross validation generation
    :param random_state: random seed for reproducibility
    :return: a vector of class prevalence values
    """

    total_prev = np.zeros(shape=data.n_classes)

    for train, test in data.kFCV(nfolds=nfolds, random_state=random_state):
        quantifier.fit(train)
        fold_prev = quantifier.quantify(test.X)
        rel_size = 1. * len(test) / len(data)
        total_prev += fold_prev*rel_size

    return total_prev


def expand_grid(param_grid: dict):
    """
    Expands a param_grid dictionary as a list of configurations.
    Example:

    >>> combinations = expand_grid({'A': [1, 10, 100], 'B': [True, False]})
    >>> print(combinations)
    >>> [{'A': 1, 'B': True}, {'A': 1, 'B': False}, {'A': 10, 'B': True}, {'A': 10, 'B': False}, {'A': 100, 'B': True}, {'A': 100, 'B': False}]

    :param param_grid: dictionary with keys representing hyper-parameter names, and values representing the range
        to explore for that hyper-parameter
    :return: a list of configurations, i.e., combinations of hyper-parameter assignments in the grid.
    """
    params_keys = list(param_grid.keys())
    params_values = list(param_grid.values())
    configs = [{k: combs[i] for i, k in enumerate(params_keys)} for combs in itertools.product(*params_values)]
    return configs


def group_params(param_grid: dict):
    """
    Partitions a param_grid dictionary as two lists of configurations, one for the classifier-specific
    hyper-parameters, and another for que quantifier-specific hyper-parameters

    :param param_grid: dictionary with keys representing hyper-parameter names, and values representing the range
        to explore for that hyper-parameter
    :return: two expanded grids of configurations, one for the classifier, another for the quantifier
    """
    classifier_params, quantifier_params = {}, {}
    for key, values in param_grid.items():
        if key.startswith('classifier__') or key == 'val_split':
            classifier_params[key] = values
        else:
            quantifier_params[key] = values

    classifier_configs = expand_grid(classifier_params)
    quantifier_configs = expand_grid(quantifier_params)

    return classifier_configs, quantifier_configs

