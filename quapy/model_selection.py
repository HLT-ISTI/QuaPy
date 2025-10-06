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


class ConfigStatus:

    def __init__(self, params, status, msg=''):
        self.params = params
        self.status = status
        self.msg = msg

    def __str__(self):
        return f':params:{self.params} :status:{self.status} ' + self.msg

    def __repr__(self):
        return str(self)

    def success(self):
        return self.status == Status.SUCCESS

    def failed(self):
        return self.status != Status.SUCCESS


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
    :param refit: whether to refit the model on the whole labelled collection (training+validation) with
        the best chosen hyperparameter combination. Ignored if protocol='gen'
    :param timeout: establishes a timer (in seconds) for each of the hyperparameters configurations being tested.
        Whenever a run takes longer than this timer, that configuration will be ignored. If all configurations end up
        being ignored, a TimeoutError exception is raised. If -1 (default) then no time bound is set.
    :param raise_errors: boolean, if True then raises an exception when a param combination yields any error, if
        otherwise is False (default), then the combination is marked with an error status, but the process goes on.
        However, if no configuration yields a valid model, then a ValueError exception will be raised.
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
                 raise_errors=False,
                 verbose=False):

        self.model = model
        self.param_grid = param_grid
        self.protocol = protocol
        self.refit = refit
        self.timeout = timeout
        self.n_jobs = qp._get_njobs(n_jobs)
        self.raise_errors = raise_errors
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

    def _prepare_classifier(self, cls_params):
        model = deepcopy(self.model)

        def job(cls_params):
            model.set_params(**cls_params)
            predictions = model.classifier_fit_predict(self._training)
            return predictions

        predictions, status, took = self._error_handler(job, cls_params)
        self._sout(f'[classifier fit] hyperparams={cls_params} [took {took:.3f}s]')
        return model, predictions, status, took

    def _prepare_aggregation(self, args):
        model, predictions, cls_took, cls_params, q_params = args
        model = deepcopy(model)
        params = {**cls_params, **q_params}

        def job(q_params):
            model.set_params(**q_params)
            model.aggregation_fit(predictions, self._training)
            score = evaluation.evaluate(model, protocol=self.protocol, error_metric=self.error)
            return score

        score, status, aggr_took = self._error_handler(job, q_params)
        self._print_status(params, score, status, aggr_took)
        return model, params, score, status, (cls_took+aggr_took)

    def _prepare_nonaggr_model(self, params):
        model = deepcopy(self.model)

        def job(params):
            model.set_params(**params)
            model.fit(self._training)
            score = evaluation.evaluate(model, protocol=self.protocol, error_metric=self.error)
            return score

        score, status, took = self._error_handler(job, params)
        self._print_status(params, score, status, took)
        return model, params, score, status, took

    def _break_down_fit(self):
        """
        Decides whether to break down the fit phase in two (classifier-fit followed by aggregation-fit).
        In order to do so, some conditions should be met: a) the quantifier is of type aggregative,
        b) the set of hyperparameters can be split into two disjoint non-empty groups.

        :return: True if the conditions are met, False otherwise
        """
        if not isinstance(self.model, AggregativeQuantifier):
            return False
        cls_configs, q_configs = group_params(self.param_grid)
        if (len(cls_configs) == 1) or (len(q_configs)==1):
            return False
        return True

    def _compute_scores_aggregative(self, training):
        # break down the set of hyperparameters into two: classifier-specific, quantifier-specific
        cls_configs, q_configs = group_params(self.param_grid)

        # train all classifiers and get the predictions
        self._training = training
        cls_outs = qp.util.parallel(
            self._prepare_classifier,
            cls_configs,
            seed=qp.environ.get('_R_SEED', None),
            n_jobs=self.n_jobs
        )

        # filter out classifier configurations that yielded any error
        success_outs = []
        for (model, predictions, status, took), cls_config in zip(cls_outs, cls_configs):
            if status.success():
                success_outs.append((model, predictions, took, cls_config))
            else:
                self.error_collector.append(status)

        if len(success_outs) == 0:
            raise ValueError('No valid configuration found for the classifier!')

        # explore the quantifier-specific hyperparameters for each valid training configuration
        aggr_configs = [(*out, q_config) for out, q_config in itertools.product(success_outs, q_configs)]
        aggr_outs = qp.util.parallel(
            self._prepare_aggregation,
            aggr_configs,
            seed=qp.environ.get('_R_SEED', None),
            n_jobs=self.n_jobs
        )

        return aggr_outs

    def _compute_scores_nonaggregative(self, training):
        configs = expand_grid(self.param_grid)
        self._training = training
        scores = qp.util.parallel(
            self._prepare_nonaggr_model,
            configs,
            seed=qp.environ.get('_R_SEED', None),
            n_jobs=self.n_jobs
        )
        return scores

    def _print_status(self, params, score, status, took):
        if status.success():
            self._sout(f'hyperparams=[{params}]\t got {self.error.__name__} = {score:.5f} [took {took:.3f}s]')
        else:
            self._sout(f'error={status}')

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

        self.error_collector = []

        self._sout(f'starting model selection with n_jobs={self.n_jobs}')
        if self._break_down_fit():
            results = self._compute_scores_aggregative(training)
        else:
            results = self._compute_scores_nonaggregative(training)

        self.param_scores_ = {}
        self.best_score_ = None
        for model, params, score, status, took in results:
            if status.success():
                if self.best_score_ is None or score < self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = params
                    self.best_model_ = model
                self.param_scores_[str(params)] = score
            else:
                self.param_scores_[str(params)] = status.status
                self.error_collector.append(status)

        tend = time()-tinit

        if self.best_score_ is None:
            raise ValueError('no combination of hyperparameters seemed to work')

        self._sout(f'optimization finished: best params {self.best_params_} (score={self.best_score_:.5f}) '
                   f'[took {tend:.4f}s]')

        no_errors = len(self.error_collector)
        if no_errors>0:
            self._sout(f'warning: {no_errors} errors found')
            for err in self.error_collector:
                self._sout(f'\t{str(err)}')

        if self.refit:
            if isinstance(self.protocol, OnLabelledCollectionProtocol):
                tinit = time()
                self._sout(f'refitting on the whole development set')
                self.best_model_.fit(training + self.protocol.get_labelled_collection())
                tend = time() - tinit
                self.refit_time_ = tend
            else:
                # already checked
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

    def _error_handler(self, func, params):
        """
        Endorses one job with two returned values: the status, and the time of execution

        :param func: the function to be called
        :param params: parameters of the function
        :return: `tuple(out, status, time)` where `out` is the function output,
            `status` is an enum value from `Status`, and `time` is the time it
            took to complete the call
        """

        output = None

        def _handle(status, exception):
            if self.raise_errors:
                raise exception
            else:
                return ConfigStatus(params, status, msg=str(exception))

        try:
            with timeout(self.timeout):
                tinit = time()
                output = func(params)
                status = ConfigStatus(params, Status.SUCCESS)

        except TimeoutError as e:
            status = _handle(Status.TIMEOUT, e)

        except ValueError as e:
            status = _handle(Status.INVALID, e)

        except Exception as e:
            status = _handle(Status.ERROR, e)

        took = time() - tinit
        return output, status, took


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

