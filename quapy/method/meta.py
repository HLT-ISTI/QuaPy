from copy import deepcopy
from typing import Union
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_predict
from tqdm import tqdm

import quapy as qp
from quapy import functional as F
from quapy.data import LabelledCollection
from quapy.evaluation import evaluate
from quapy.model_selection import GridSearchQ

try:
    from . import neural
except ModuleNotFoundError:
    neural = None
from .base import BaseQuantifier
from quapy.method.aggregative import CC, ACC, PACC, HDy, EMQ

if neural:
    QuaNet = neural.QuaNetTrainer
else:
    QuaNet = "QuaNet is not available due to missing torch package"


class Ensemble(BaseQuantifier):
    VALID_POLICIES = {'ave', 'ptr', 'ds'} | qp.error.QUANTIFICATION_ERROR_NAMES

    """
    Implementation of the Ensemble methods for quantification described by 
    `Pérez-Gállego et al., 2017 <https://www.sciencedirect.com/science/article/pii/S1566253516300628>`_
    and
    `Pérez-Gállego et al., 2019 <https://www.sciencedirect.com/science/article/pii/S1566253517303652>`_.
    The policies implemented include:
    
    - Average (`policy='ave'`): computes class prevalence estimates as the average of the estimates 
      returned by the base quantifiers.
    - Training Prevalence (`policy='ptr'`): applies a dynamic selection to the ensemble’s members by retaining only 
      those members such that the class prevalence values in the samples they use as training set are closest to 
      preliminary class prevalence estimates computed as the average of the estimates of all the members. The final 
      estimate is recomputed by considering only the selected members.
    - Distribution Similarity (`policy='ds'`): performs a dynamic selection of base members by retaining
      the members trained on samples whose distribution of posterior probabilities is closest, in terms of the
      Hellinger Distance, to the distribution of posterior probabilities in the test sample
    - Accuracy (`policy='<valid error name>'`): performs a static selection of the ensemble members by
      retaining those that minimize a quantification error measure, which is passed as an argument.
      
    Example:
    
    >>> model = Ensemble(quantifier=ACC(LogisticRegression()), size=30, policy='ave', n_jobs=-1)
    
    :param quantifier: base quantification member of the ensemble 
    :param size: number of members
    :param red_size: number of members to retain after selection (depending on the policy)
    :param min_pos: minimum number of positive instances to consider a sample as valid 
    :param policy: the selection policy; available policies include: `ave` (default), `ptr`, `ds`, and accuracy 
        (which is instantiated via a valid error name, e.g., `mae`)
    :param max_sample_size: maximum number of instances to consider in the samples (set to None 
        to indicate no limit, default)
    :param val_split: a float in range (0,1) indicating the proportion of data to be used as a stratified held-out
        validation split, or a :class:`quapy.data.base.LabelledCollection` (the split itself).
    :param n_jobs: number of parallel workers (default 1)
    :param verbose: set to True (default is False) to get some information in standard output
    """

    def __init__(self,
                 quantifier: BaseQuantifier,
                 size=50,
                 red_size=25,
                 min_pos=5,
                 policy='ave',
                 max_sample_size=None,
                 val_split:Union[qp.data.LabelledCollection, float]=None,
                 n_jobs=1,
                 verbose=False):
        assert policy in Ensemble.VALID_POLICIES, \
            f'unknown policy={policy}; valid are {Ensemble.VALID_POLICIES}'
        assert max_sample_size is None or max_sample_size > 0, \
            'wrong value for max_sample_size; set it to a positive number or None'
        self.base_quantifier = quantifier
        self.size = size
        self.min_pos = min_pos
        self.red_size = red_size
        self.policy = policy
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.post_proba_fn = None
        self.verbose = verbose
        self.max_sample_size = max_sample_size

    def _sout(self, msg):
        if self.verbose:
            print('[Ensemble]' + msg)

    def fit(self, data: qp.data.LabelledCollection, val_split: Union[qp.data.LabelledCollection, float] = None):
        self._sout('Fit')
        if self.policy == 'ds' and not data.binary:
            raise ValueError(f'ds policy is only defined for binary quantification, but this dataset is not binary')
        if val_split is None:
            val_split = self.val_split

        # randomly chooses the prevalences for each member of the ensemble (preventing classes with less than
        # min_pos positive examples)
        sample_size = len(data) if self.max_sample_size is None else min(self.max_sample_size, len(data))
        prevs = [_draw_simplex(ndim=data.n_classes, min_val=self.min_pos / sample_size) for _ in range(self.size)]

        posteriors = None
        if self.policy == 'ds':
            # precompute the training posterior probabilities
            posteriors, self.post_proba_fn = self._ds_policy_get_posteriors(data)

        is_static_policy = (self.policy in qp.error.QUANTIFICATION_ERROR_NAMES)

        args = (
            (self.base_quantifier, data, val_split, prev, posteriors, is_static_policy, self.verbose, sample_size)
            for prev in prevs
        )
        self.ensemble = qp.util.parallel(
            _delayed_new_instance,
            tqdm(args, desc='fitting ensamble', total=self.size) if self.verbose else args,
            n_jobs=self.n_jobs)

        # static selection policy (the name of a quantification-oriented error function to minimize)
        if self.policy in qp.error.QUANTIFICATION_ERROR_NAMES:
            self._accuracy_policy(error_name=self.policy)

        self._sout('Fit [Done]')
        return self

    def quantify(self, instances):
        predictions = np.asarray(
            qp.util.parallel(_delayed_quantify, ((Qi, instances) for Qi in self.ensemble), n_jobs=self.n_jobs)
        )

        if self.policy == 'ptr':
            predictions = self._ptr_policy(predictions)
        elif self.policy == 'ds':
            predictions = self._ds_policy(predictions, instances)

        predictions = np.mean(predictions, axis=0)
        return F.normalize_prevalence(predictions)

    def set_params(self, **parameters):
        """
        This function should not be used within :class:`quapy.model_selection.GridSearchQ` (is here for compatibility
        with the abstract class).
        Instead, use `Ensemble(GridSearchQ(q),...)`, with `q` a Quantifier (recommended), or
        `Ensemble(Q(GridSearchCV(l)))` with `Q` a quantifier class that has a learner `l` optimized for
         classification (not recommended).

        :param parameters: dictionary
        :return: raises an Exception
        """
        raise NotImplementedError(f'{self.__class__.__name__} should not be used within GridSearchQ; '
                                  f'instead, use Ensemble(GridSearchQ(q),...), with q a Quantifier (recommended), '
                                  f'or Ensemble(Q(GridSearchCV(l))) with Q a quantifier class that has a learner '
                                  f'l optimized for classification (not recommended).')

    def get_params(self, deep=True):
        """
        This function should not be used within :class:`quapy.model_selection.GridSearchQ` (is here for compatibility
        with the abstract class).
        Instead, use `Ensemble(GridSearchQ(q),...)`, with `q` a Quantifier (recommended), or
        `Ensemble(Q(GridSearchCV(l)))` with `Q` a quantifier class that has a learner `l` optimized for
         classification (not recommended).

        :return: raises an Exception
        """
        raise NotImplementedError()

    def _accuracy_policy(self, error_name):
        """
        Selects the red_size best performant quantifiers in a static way (i.e., dropping all non-selected instances).
        For each model in the ensemble, the performance is measured in terms of _error_name_ on the quantification of
        the samples used for training the rest of the models in the ensemble.
        """
        error = qp.error.from_name(error_name)
        tests = [m[3] for m in self.ensemble]
        scores = []
        for i, model in enumerate(self.ensemble):
            scores.append(evaluate(model[0], tests[:i] + tests[i + 1:], error, self.n_jobs))
        order = np.argsort(scores)

        self.ensemble = _select_k(self.ensemble, order, k=self.red_size)

    def _ptr_policy(self, predictions):
        """
        Selects the predictions made by models that have been trained on samples with a prevalence that is most similar
        to a first approximation of the test prevalence as made by all models in the ensemble.
        """
        test_prev_estim = predictions.mean(axis=0)
        tr_prevs = [m[1] for m in self.ensemble]
        ptr_differences = [qp.error.mse(ptr_i, test_prev_estim) for ptr_i in tr_prevs]
        order = np.argsort(ptr_differences)
        return _select_k(predictions, order, k=self.red_size)

    def _ds_policy_get_posteriors(self, data: LabelledCollection):
        """
        In the original article, this procedure is not described in a sufficient level of detail. The paper only says
        that the distribution of posterior probabilities from training and test examples is compared by means of the
        Hellinger Distance. However, how these posterior probabilities are generated is not specified. In the article,
        a Logistic Regressor (LR) is used as the classifier device and that could be used for this purpose. However, in
        general, a Quantifier is not necessarily an instance of Aggreggative Probabilistic Quantifiers, and so, that the
        quantifier builds on top of a probabilistic classifier cannot be given for granted. Additionally, it would not
        be correct to generate the posterior probabilities for training documents that have concurred in training the
        classifier that generates them.
        This function thus generates the posterior probabilities for all training documents in a cross-validation way,
        using a LR with hyperparameters that have previously been optimized via grid search in 5FCV.
        :return P,f, where P is a ndarray containing the posterior probabilities of the training data, generated via
        cross-validation and using an optimized LR, and the function to be used in order to generate posterior
        probabilities for test instances.
        """
        X, y = data.Xy
        lr_base = LogisticRegression(class_weight='balanced', max_iter=1000)

        optim = GridSearchCV(
            lr_base, param_grid={'C': np.logspace(-4, 4, 9)}, cv=5, n_jobs=self.n_jobs, refit=True
        ).fit(X, y)

        posteriors = cross_val_predict(
            optim.best_estimator_, X, y, cv=5, n_jobs=self.n_jobs, method='predict_proba'
        )
        posteriors_generator = optim.best_estimator_.predict_proba

        return posteriors, posteriors_generator

    def _ds_policy(self, predictions, test):
        test_posteriors = self.post_proba_fn(test)
        test_distribution = get_probability_distribution(test_posteriors)
        tr_distributions = [m[2] for m in self.ensemble]
        dist = [F.HellingerDistance(tr_dist_i, test_distribution) for tr_dist_i in tr_distributions]
        order = np.argsort(dist)
        return _select_k(predictions, order, k=self.red_size)

    @property
    def classes_(self):
        return self.base_quantifier.classes_

    @property
    def binary(self):
        """
        Returns a boolean indicating whether the base quantifiers are binary or not

        :return: boolean
        """
        return self.base_quantifier.binary

    @property
    def aggregative(self):
        """
        Indicates that the quantifier is not aggregative.

        :return: False
        """
        return False

    @property
    def probabilistic(self):
        """
        Indicates that the quantifier is not probabilistic.

        :return: False
        """
        return False


def get_probability_distribution(posterior_probabilities, bins=8):
    """
    Gets a histogram out of the posterior probabilities (only for the binary case).

    :param posterior_probabilities: array-like of shape `(n_instances, 2,)`
    :param bins: integer
    :return: `np.ndarray` with the relative frequencies for each bin (for the positive class only)
    """
    assert posterior_probabilities.shape[1] == 2, 'the posterior probabilities do not seem to be for a binary problem'
    posterior_probabilities = posterior_probabilities[:, 1]  # take the positive posteriors only
    distribution, _ = np.histogram(posterior_probabilities, bins=bins, range=(0, 1), density=True)
    return distribution


def _select_k(elements, order, k):
    return [elements[idx] for idx in order[:k]]


def _delayed_new_instance(args):
    base_quantifier, data, val_split, prev, posteriors, keep_samples, verbose, sample_size = args
    if verbose:
        print(f'\tfit-start for prev {F.strprev(prev)}, sample_size={sample_size}')
    model = deepcopy(base_quantifier)

    if val_split is not None:
        if isinstance(val_split, float):
            assert 0 < val_split < 1, 'val_split should be in (0,1)'
            data, val_split = data.split_stratified(train_prop=1 - val_split)

    sample_index = data.sampling_index(sample_size, *prev)
    sample = data.sampling_from_index(sample_index)

    if val_split is not None:
        model.fit(sample, val_split=val_split)
    else:
        model.fit(sample)

    tr_prevalence = sample.prevalence()
    tr_distribution = get_probability_distribution(posteriors[sample_index]) if (posteriors is not None) else None
    if verbose:
        print(f'\t\--fit-ended for prev {F.strprev(prev)}')
    return (model, tr_prevalence, tr_distribution, sample if keep_samples else None)


def _delayed_quantify(args):
    quantifier, instances = args
    return quantifier[0].quantify(instances)


def _draw_simplex(ndim, min_val, max_trials=100):
    """
    returns a uniform sampling from the ndim-dimensional simplex but guarantees that all dimensions
    are >= min_class_prev (for min_val>0, this makes the sampling not truly uniform)
    :param ndim: number of dimensions of the simplex
    :param min_val: minimum class prevalence allowed. If less than 1/ndim a ValueError will be throw since
    there is no possible solution.
    :return: a sample from the ndim-dimensional simplex that is uniform in S(ndim)-R where S(ndim) is the simplex
    and R is the simplex subset containing dimensions lower than min_val
    """
    if min_val >= 1 / ndim:
        raise ValueError(f'no sample can be draw from the {ndim}-dimensional simplex so that '
                         f'all its values are >={min_val} (try with a larger value for min_pos)')
    trials = 0
    while True:
        u = F.uniform_simplex_sampling(ndim)
        if all(u >= min_val):
            return u
        trials += 1
        if trials >= max_trials:
            raise ValueError(f'it looks like finding a random simplex with all its dimensions being'
                             f'>= {min_val} is unlikely (it failed after {max_trials} trials)')


def _instantiate_ensemble(learner, base_quantifier_class, param_grid, optim, param_model_sel, **kwargs):
    if optim is None:
        base_quantifier = base_quantifier_class(learner)
    elif optim in qp.error.CLASSIFICATION_ERROR:
        if optim == qp.error.f1e:
            scoring = make_scorer(f1_score)
        elif optim == qp.error.acce:
            scoring = make_scorer(accuracy_score)
        learner = GridSearchCV(learner, param_grid, scoring=scoring)
        base_quantifier = base_quantifier_class(learner)
    else:
        base_quantifier = GridSearchQ(base_quantifier_class(learner),
                                      param_grid=param_grid,
                                      **param_model_sel,
                                      error=optim)

    return Ensemble(base_quantifier, **kwargs)


def _check_error(error):
    if error is None:
        return None
    if error in qp.error.QUANTIFICATION_ERROR or error in qp.error.CLASSIFICATION_ERROR:
        return error
    elif isinstance(error, str):
        return qp.error.from_name(error)
    else:
        raise ValueError(f'unexpected error type; must either be a callable function or a str representing\n'
                         f'the name of an error function in {qp.error.ERROR_NAMES}')


def ensembleFactory(learner, base_quantifier_class, param_grid=None, optim=None, param_model_sel: dict = None,
                    **kwargs):
    """
    Ensemble factory. Provides a unified interface for instantiating ensembles that can be optimized (via model
    selection for quantification) for a given evaluation metric using :class:`quapy.model_selection.GridSearchQ`.
    If the evaluation metric is classification-oriented
    (instead of quantification-oriented), then the optimization will be carried out via sklearn's
    `GridSearchCV <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_.

    Example to instantiate an :class:`Ensemble` based on :class:`quapy.method.aggregative.PACC`
    in which the base members are optimized for :meth:`quapy.error.mae` via
    :class:`quapy.model_selection.GridSearchQ`. The ensemble follows the policy `Accuracy` based
    on :meth:`quapy.error.mae` (the same measure being optimized),
    meaning that a static selection of members of the ensemble is made based on their performance
    in terms of this error.

    >>> param_grid = {
    >>>     'C': np.logspace(-3,3,7),
    >>>     'class_weight': ['balanced', None]
    >>> }
    >>> param_mod_sel = {
    >>>     'sample_size': 500,
    >>>     'protocol': 'app'
    >>> }
    >>> common={
    >>>     'max_sample_size': 1000,
    >>>     'n_jobs': -1,
    >>>     'param_grid': param_grid,
    >>>     'param_mod_sel': param_mod_sel,
    >>> }
    >>>
    >>> ensembleFactory(LogisticRegression(), PACC, optim='mae', policy='mae', **common)

    :param learner: sklearn's Estimator that generates a classifier
    :param base_quantifier_class: a class of quantifiers
    :param param_grid: a dictionary with the grid of parameters to optimize for
    :param optim: a valid quantification or classification error, or a string name of it
    :param param_model_sel: a dictionary containing any keyworded argument to pass to
        :class:`quapy.model_selection.GridSearchQ`
    :param kwargs: kwargs for the class :class:`Ensemble`
    :return: an instance of :class:`Ensemble`
    """
    if optim is not None:
        if param_grid is None:
            raise ValueError(f'param_grid is None but optim was requested.')
        if param_model_sel is None:
            raise ValueError(f'param_model_sel is None but optim was requested.')
    error = _check_error(optim)
    return _instantiate_ensemble(learner, base_quantifier_class, param_grid, error, param_model_sel, **kwargs)


def ECC(learner, param_grid=None, optim=None, param_mod_sel=None, **kwargs):
    """
    Implements an ensemble of :class:`quapy.method.aggregative.CC` quantifiers, as used by
    `Pérez-Gállego et al., 2019 <https://www.sciencedirect.com/science/article/pii/S1566253517303652>`_.

    Equivalent to:

    >>> ensembleFactory(learner, CC, param_grid, optim, param_mod_sel, **kwargs)

    See :meth:`ensembleFactory` for further details.

    :param learner: sklearn's Estimator that generates a classifier
    :param param_grid: a dictionary with the grid of parameters to optimize for
    :param optim: a valid quantification or classification error, or a string name of it
    :param param_model_sel: a dictionary containing any keyworded argument to pass to
        :class:`quapy.model_selection.GridSearchQ`
    :param kwargs: kwargs for the class :class:`Ensemble`
    :return: an instance of :class:`Ensemble`
    """

    return ensembleFactory(learner, CC, param_grid, optim, param_mod_sel, **kwargs)


def EACC(learner, param_grid=None, optim=None, param_mod_sel=None, **kwargs):
    """
    Implements an ensemble of :class:`quapy.method.aggregative.ACC` quantifiers, as used by
    `Pérez-Gállego et al., 2019 <https://www.sciencedirect.com/science/article/pii/S1566253517303652>`_.

    Equivalent to:

    >>> ensembleFactory(learner, ACC, param_grid, optim, param_mod_sel, **kwargs)

    See :meth:`ensembleFactory` for further details.

    :param learner: sklearn's Estimator that generates a classifier
    :param param_grid: a dictionary with the grid of parameters to optimize for
    :param optim: a valid quantification or classification error, or a string name of it
    :param param_model_sel: a dictionary containing any keyworded argument to pass to
        :class:`quapy.model_selection.GridSearchQ`
    :param kwargs: kwargs for the class :class:`Ensemble`
    :return: an instance of :class:`Ensemble`
    """

    return ensembleFactory(learner, ACC, param_grid, optim, param_mod_sel, **kwargs)


def EPACC(learner, param_grid=None, optim=None, param_mod_sel=None, **kwargs):
    """
    Implements an ensemble of :class:`quapy.method.aggregative.PACC` quantifiers.

    Equivalent to:

    >>> ensembleFactory(learner, PACC, param_grid, optim, param_mod_sel, **kwargs)

    See :meth:`ensembleFactory` for further details.

    :param learner: sklearn's Estimator that generates a classifier
    :param param_grid: a dictionary with the grid of parameters to optimize for
    :param optim: a valid quantification or classification error, or a string name of it
    :param param_model_sel: a dictionary containing any keyworded argument to pass to
        :class:`quapy.model_selection.GridSearchQ`
    :param kwargs: kwargs for the class :class:`Ensemble`
    :return: an instance of :class:`Ensemble`
    """

    return ensembleFactory(learner, PACC, param_grid, optim, param_mod_sel, **kwargs)


def EHDy(learner, param_grid=None, optim=None, param_mod_sel=None, **kwargs):
    """
    Implements an ensemble of :class:`quapy.method.aggregative.HDy` quantifiers, as used by
    `Pérez-Gállego et al., 2019 <https://www.sciencedirect.com/science/article/pii/S1566253517303652>`_.

    Equivalent to:

    >>> ensembleFactory(learner, HDy, param_grid, optim, param_mod_sel, **kwargs)

    See :meth:`ensembleFactory` for further details.

    :param learner: sklearn's Estimator that generates a classifier
    :param param_grid: a dictionary with the grid of parameters to optimize for
    :param optim: a valid quantification or classification error, or a string name of it
    :param param_model_sel: a dictionary containing any keyworded argument to pass to
        :class:`quapy.model_selection.GridSearchQ`
    :param kwargs: kwargs for the class :class:`Ensemble`
    :return: an instance of :class:`Ensemble`
    """

    return ensembleFactory(learner, HDy, param_grid, optim, param_mod_sel, **kwargs)


def EEMQ(learner, param_grid=None, optim=None, param_mod_sel=None, **kwargs):
    """
    Implements an ensemble of :class:`quapy.method.aggregative.EMQ` quantifiers.

    Equivalent to:

    >>> ensembleFactory(learner, EMQ, param_grid, optim, param_mod_sel, **kwargs)

    See :meth:`ensembleFactory` for further details.

    :param learner: sklearn's Estimator that generates a classifier
    :param param_grid: a dictionary with the grid of parameters to optimize for
    :param optim: a valid quantification or classification error, or a string name of it
    :param param_model_sel: a dictionary containing any keyworded argument to pass to
        :class:`quapy.model_selection.GridSearchQ`
    :param kwargs: kwargs for the class :class:`Ensemble`
    :return: an instance of :class:`Ensemble`
    """

    return ensembleFactory(learner, EMQ, param_grid, optim, param_mod_sel, **kwargs)
