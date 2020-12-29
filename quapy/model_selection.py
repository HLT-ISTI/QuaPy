import itertools
import quapy as qp
from evaluation import artificial_sampling_prediction
from data.base import LabelledCollection
from method.aggregative import BaseQuantifier
from typing import Union, Callable
import quapy.functional as F
from copy import deepcopy


class GridSearchQ:

    def __init__(self,
                 model : BaseQuantifier,
                 param_grid: dict,
                 sample_size: int,
                 n_prevpoints: int = None,
                 n_repetitions: int = 1,
                 eval_budget : int = None,
                 error: Union[Callable, str] = qp.error.mae,
                 refit=False,
                 n_jobs=-1,
                 random_seed=42,
                 verbose=False):
        """
        Optimizes the hyperparameters of a quantification method, based on an evaluation method and on an evaluation
        protocol for quantification.
        :param model: the quantifier to optimize
        :param training: the training set on which to optimize the hyperparameters
        :param validation: either a LabelledCollection on which to test the performance of the different settings, or
        a float in [0,1] indicating the proportion of labelled data to extract from the training set
        :param param_grid: a dictionary with keys the parameter names and values the list of values to explore for
        that particular parameter
        :param sample_size: the size of the samples to extract from the validation set
        :param n_prevpoints: if specified, indicates the number of equally distant point to extract from the interval
        [0,1] in order to define the prevalences of the samples; e.g., if n_prevpoints=5, then the prevalences for
        each class will be explored in [0.00, 0.25, 0.50, 0.75, 1.00]. If not specified, then eval_budget is requested
        :param n_repetitions: the number of repetitions for each combination of prevalences. This parameter is ignored
        if eval_budget is set and is lower than the number of combinations that would be generated using the value
        assigned to n_prevpoints (for the current number of classes and n_repetitions)
        :param eval_budget: if specified, sets a ceil on the number of evaluations to perform for each hyper-parameter
        combination. For example, if there are 3 classes, n_repetitions=1 and eval_budget=20, then n_prevpoints will be
        set to 5, since this will generate 15 different prevalences:
         [0, 0, 1], [0, 0.25, 0.75], [0, 0.5, 0.5] ... [1, 0, 0]
        :param error: an error function (callable) or a string indicating the name of an error function (valid ones
        are those in qp.error.QUANTIFICATION_ERROR
        :param refit: whether or not to refit the model on the whole labelled collection (training+validation) with
        the best chosen hyperparameter combination
        :param n_jobs: number of parallel jobs
        :param random_seed: set the seed of the random generator to replicate experiments
        :param verbose: set to True to get information through the stdout
        """
        self.model = model
        self.param_grid = param_grid
        self.sample_size = sample_size
        self.n_prevpoints = n_prevpoints
        self.n_repetitions = n_repetitions
        self.eval_budget = eval_budget
        self.refit = refit
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.verbose = verbose

        self.__check_error(error)

    def sout(self, msg):
        if self.verbose:
            print(f'[{self.__class__.__name__}]: {msg}')

    def __check_training_validation(self, training, validation):
        if isinstance(validation, LabelledCollection):
            return training, validation
        elif isinstance(validation, float):
            assert 0. < validation < 1., 'validation proportion should be in (0,1)'
            training, validation = training.split_stratified(train_prop=1-validation)
            return training, validation
        else:
            raise ValueError('"validation" must either be a LabelledCollection or a float in (0,1) indicating the'
                             'proportion of training documents to extract')

    def __check_num_evals(self, n_prevpoints, eval_budget, n_repetitions, n_classes):
        if n_prevpoints is None and eval_budget is None:
            raise ValueError('either n_prevpoints or eval_budget has to be specified')
        elif n_prevpoints is None:
            assert eval_budget > 0, 'eval_budget must be a positive integer'
            self.n_prevpoints = F.get_nprevpoints_approximation(eval_budget, n_classes, n_repetitions)
            eval_computations = F.num_prevalence_combinations(self.n_prevpoints, n_classes, n_repetitions)
            self.sout(f'setting n_prevpoints={self.n_prevpoints} so that the number of \n'
                  f'evaluations ({eval_computations}) does not exceed the evaluation budget ({eval_budget})')
        elif eval_budget is None:
            self.n_prevpoints = n_prevpoints
            eval_computations = F.num_prevalence_combinations(self.n_prevpoints, n_classes, n_repetitions)
            self.sout(f'{eval_computations} evaluations will be performed for each\n'
                  f'combination of hyper-parameters')
        else:
            eval_computations = F.num_prevalence_combinations(n_prevpoints, n_classes, n_repetitions)
            if eval_computations > eval_budget:
                self.n_prevpoints = F.get_nprevpoints_approximation(eval_budget, n_classes, n_repetitions)
                new_eval_computations = F.num_prevalence_combinations(self.n_prevpoints, n_classes, n_repetitions)
                self.sout(f'the budget of evaluations would be exceeded with\n'
                      f'n_prevpoints={n_prevpoints}. Chaning to n_prevpoints={self.n_prevpoints}. This will produce\n'
                      f'{new_eval_computations} evaluation computations for each hyper-parameter combination.')

    def __check_error(self, error):
        if error in qp.error.QUANTIFICATION_ERROR:
            self.error = error
        elif isinstance(error, str):
            assert error in {func.__name__ for func in qp.error.QUANTIFICATION_ERROR}, \
                f'unknown error name; valid ones are {qp.error.QUANTIFICATION_ERROR}'
            self.error = getattr(qp.error, error)
        else:
            raise ValueError(f'unexpected error type; must either be a callable function or a str representing\n'
                             f'the name of an error function in {qp.error.QUANTIFICATION_ERROR}')

    def fit(self, training: LabelledCollection, validation: Union[LabelledCollection, float]):
        """
        :param training: the training set on which to optimize the hyperparameters
        :param validation: either a LabelledCollection on which to test the performance of the different settings, or
        a float in [0,1] indicating the proportion of labelled data to extract from the training set
        """
        training, validation = self.__check_training_validation(training, validation)
        self.__check_num_evals(self.n_prevpoints, self.eval_budget, self.n_repetitions, training.n_classes)

        params_keys = list(self.param_grid.keys())
        params_values = list(self.param_grid.values())

        model = self.model
        n_jobs = self.n_jobs

        self.sout(f'starting optimization with n_jobs={n_jobs}')
        self.param_scores_ = {}
        self.best_score_ = None
        for values in itertools.product(*params_values):
            params = {k: values[i] for i, k in enumerate(params_keys)}

            # overrides default parameters with the parameters being explored at this iteration
            model.set_params(**params)
            model.fit(training)
            true_prevalences, estim_prevalences = artificial_sampling_prediction(
                model, validation, self.sample_size, self.n_prevpoints, self.n_repetitions, n_jobs, self.random_seed,
                verbose=False
            )

            score = self.error(true_prevalences, estim_prevalences)
            self.sout(f'checking hyperparams={params} got {self.error.__name__} score {score:.5f}')
            if self.best_score_ is None or score < self.best_score_:
                self.best_score_ = score
                self.best_params_ = params
                if not self.refit:
                    self.best_model_ = deepcopy(model)
            self.param_scores_[str(params)] = score

        self.sout(f'optimization finished: best params {self.best_params_} (score={self.best_score_:.5f})')
        model.set_params(**self.best_params_)
        self.best_model_ = deepcopy(model)

        if self.refit:
            self.sout(f'refitting on the whole development set')
            self.best_model_.fit(training + validation)

        return self.best_model_

