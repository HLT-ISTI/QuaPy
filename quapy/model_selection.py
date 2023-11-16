import itertools
import signal
from copy import deepcopy
from enum import Enum
from typing import Union, Callable

import numpy as np
from sklearn import clone

import quapy as qp
from quapy import evaluation
from quapy.protocol import AbstractProtocol, OnLabelledCollectionProtocol
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import BaseQuantifier, AggregativeQuantifier
from time import time


class Status(Enum):
    SUCCESS = 1
    TIMEOUT = 2
    INVALID = 3
    ERROR = 4

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

    def _fit_nonaggregative(self, training):
        configs = expand_grid(self.param_grid)

        self._sout(f'starting model selection with {self.n_jobs =}')
        #pass a seed to parallel so it is set in child processes
        scores = qp.util.parallel(
            self._delayed_eval,
            ((params, training) for params in configs),
            seed=qp.environ.get('_R_SEED', None),
            n_jobs=self.n_jobs
        )
        return scores

    def _delayed_fit_classifier(self, args):
        cls_params, training = args
        model = deepcopy(self.model)
        model.set_params(**cls_params)
        predictions = model.classifier_fit_predict(training)
        return (model, predictions, cls_params)

    def _eval_aggregative(self, args):
        ((model, predictions, cls_params), q_params), training = args
        model = deepcopy(model)
        # overrides default parameters with the parameters being explored at this iteration
        model.set_params(**q_params)
        model.aggregation_fit(predictions, training)
        params = {**cls_params, **q_params}
        return model, params

    def _delayed_evaluation__(self, args):

        exit_status = Status.SUCCESS

        tinit = time()
        if self.timeout > 0:
            def handler(signum, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(self.timeout)

        try:
            model, params = self._eval_aggregative(args)

            score = evaluation.evaluate(model, protocol=self.protocol, error_metric=self.error)

            ttime = time() - tinit
            self._sout(f'hyperparams=[{params}]\t got {self.error.__name__} score {score:.5f} [took {ttime:.4f}s]')

            if self.timeout > 0:
                signal.alarm(0)

        except TimeoutError:
            self._sout(f'timeout ({self.timeout}s) reached for config {params}')
            score = None
            exit_status = Status.TIMEOUT

        except ValueError as e:
            self._sout(f'the combination of hyperparameters {params} is invalid')
            score = None
            exit_status = Status.INVALID

        except Exception as e:
            self._sout(f'something went wrong for config {params}; skipping:')
            self._sout(f'\tException: {e}')
            score = None
            exit_status = Status.ERROR


        return params, score, model, exit_status

    # def _delayed_fit_aggregation_and_eval(self, args):
    #
    #     ((model, predictions, cls_params), q_params), training = args
    #     exit_status = Status.SUCCESS
    #
    #     tinit = time()
    #     if self.timeout > 0:
    #         def handler(signum, frame):
    #             raise TimeoutError()
    #         signal.signal(signal.SIGALRM, handler)
    #         signal.alarm(self.timeout)
    #
    #     try:
    #         model = deepcopy(model)
    #         # overrides default parameters with the parameters being explored at this iteration
    #         model.set_params(**q_params)
    #         model.aggregation_fit(predictions, training)
    #         score = evaluation.evaluate(model, protocol=self.protocol, error_metric=self.error)
    #
    #         ttime = time() - tinit
    #         self._sout(f'hyperparams=[cls:{cls_params}, q:{q_params}]\t got {self.error.__name__} score {score:.5f} [took {ttime:.4f}s]')
    #
    #         if self.timeout > 0:
    #             signal.alarm(0)
    #     except TimeoutError:
    #         self._sout(f'timeout ({self.timeout}s) reached for config {q_params}')
    #         score = None
    #         exit_status = Status.TIMEOUT
    #     except ValueError as e:
    #         self._sout(f'the combination of hyperparameters {q_params} is invalid')
    #         score = None
    #         exit_status = Status.INVALID
    #     except Exception as e:
    #         self._sout(f'something went wrong for config {q_params}; skipping:')
    #         self._sout(f'\tException: {e}')
    #         score = None
    #         exit_status = Status.ERROR
    #
    #     params = {**cls_params, **q_params}
    #     return params, score, model, exit_status

    def _delayed_eval(self, args):
        params, training = args

        protocol = self.protocol
        error = self.error

        if self.timeout > 0:
            def handler(signum, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, handler)

        tinit = time()

        if self.timeout > 0:
            signal.alarm(self.timeout)

        try:
            model = deepcopy(self.model)
            # overrides default parameters with the parameters being explored at this iteration
            model.set_params(**params)
            model.fit(training)
            score = evaluation.evaluate(model, protocol=protocol, error_metric=error)

            ttime = time()-tinit
            self._sout(f'hyperparams={params}\t got {error.__name__} score {score:.5f} [took {ttime:.4f}s]')

            if self.timeout > 0:
                signal.alarm(0)
        except TimeoutError:
            self._sout(f'timeout ({self.timeout}s) reached for config {params}')
            score = None
        except ValueError as e:
            self._sout(f'the combination of hyperparameters {params} is invalid')
            raise e
        except Exception as e:
            self._sout(f'something went wrong for config {params}; skipping:')
            self._sout(f'\tException: {e}')
            score = None

        return params, score, model, status

    def _fit_aggregative(self, training):

        # break down the set of hyperparameters into two: classifier-specific, quantifier-specific
        cls_configs, q_configs = group_params(self.param_grid)

        # train all classifiers and get the predictions
        models_preds_clsconfigs = qp.util.parallel(
            self._delayed_fit_classifier,
            ((params, training) for params in cls_configs),
            seed=qp.environ.get('_R_SEED', None),
            n_jobs=self.n_jobs,
            asarray=False,
        )

        # explore the quantifier-specific hyperparameters for each training configuration
        scores = qp.util.parallel(
            self._delayed_fit_aggregation_and_eval,
            ((setup, training) for setup in itertools.product(models_preds_clsconfigs, q_configs)),
            seed=qp.environ.get('_R_SEED', None),
            n_jobs=self.n_jobs
        )

        return scores


    def fit(self, training: LabelledCollection):
        """ Learning routine. Fits methods with all combinations of hyperparameters and selects the one minimizing
            the error metric.

        :param training: the training set on which to optimize the hyperparameters
        :return: self
        """

        if self.refit and not isinstance(self.protocol, OnLabelledCollectionProtocol):
                raise RuntimeWarning(f'"refit" was requested, but the protocol does not '
                                     f'implement the {OnLabelledCollectionProtocol.__name__} interface')

        tinit = time()

        if isinstance(self.model, AggregativeQuantifier):
            self.results = self._fit_aggregative(training)
        else:
            self.results = self._fit_nonaggregative(training)

        self.param_scores_ = {}
        self.best_score_ = None
        for params, score, model in self.results:
            if score is not None:
                if self.best_score_ is None or score < self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = params
                    self.best_model_ = model
                self.param_scores_[str(params)] = score
            else:
                self.param_scores_[str(params)] = 'timeout'

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

