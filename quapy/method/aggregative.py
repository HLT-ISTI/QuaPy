from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Literal, Union
import numpy as np
from abstention.calibration import NoBiasVectorScaling, TempScaling, VectorScaling
from numpy.f2py.crackfortran import true_intent_list
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.utils.validation import check_is_fitted

import quapy as qp
import quapy.functional as F
from quapy.functional import get_divergence
from quapy.classification.svmperf import SVMperf
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier, BinaryQuantifier, OneVsAllGeneric
from quapy.method import _bayesian


# Abstract classes
# ------------------------------------

class AggregativeQuantifier(BaseQuantifier, ABC):
    """
    Abstract class for quantification methods that base their estimations on the aggregation of classification
    results. Aggregative quantifiers implement a pipeline that consists of generating classification predictions
    and aggregating them. For this reason, the training phase is implemented by :meth:`classification_fit` followed
    by :meth:`aggregation_fit`, while the testing phase is implemented by :meth:`classify` followed by
    :meth:`aggregate`. Subclasses of this abstract class must provide implementations for these methods.
    Aggregative quantifiers also maintain a :attr:`classifier` attribute.

    The method :meth:`fit` comes with a default implementation based on :meth:`classification_fit`
    and :meth:`aggregation_fit`.

    The method :meth:`quantify` comes with a default implementation based on :meth:`classify`
    and :meth:`aggregate`.

    :param classifier: a scikit-learn's BaseEstimator, or None, in which case the classifier is taken to be
        the one indicated in `qp.environ['DEFAULT_CLS']`
    :param fit_classifier: whether to train the learner (default is True). Set to False if the
        learner has been trained outside the quantifier.
    :param val_split: specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer (default 5), indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`); or as a tuple `(X,y)` defining the specific set of data to use for validation. Set to
        None when the method does not require any validation data, in order to avoid that some portion of
        the training data be wasted.
    """

    def __init__(self, classifier: Union[None,BaseEstimator], fit_classifier:bool=True, val_split:Union[int,float,tuple,None]=5):
        self.classifier = qp._get_classifier(classifier)
        self.fit_classifier = fit_classifier
        self.val_split = val_split

        # basic type checks
        assert hasattr(self.classifier, 'fit'), \
            f'the classifier does not implement "fit"'

        assert isinstance(fit_classifier, bool), \
            f'unexpected type for {fit_classifier=}; must be True or False'

        if isinstance(val_split, int):
            assert val_split > 1, \
                (f'when {val_split=} is indicated as an integer, it represents the number of folds in a kFCV '
                 f'and must thus be >1')
            if val_split==5 and not fit_classifier:
                print(f'Warning: {val_split=} will be ignored when the classifier is already trained '
                      f'({fit_classifier=}). Parameter {self.val_split=} will be set to None. Set {val_split=} '
                      f'to None to avoid this warning.')
                self.val_split=None
            if val_split!=5:
                assert fit_classifier, (f'Parameter {val_split=} has been modified, but {fit_classifier=} '
                                        f'indicates the classifier should not be retrained.')
        elif isinstance(val_split, float):
            assert 0 < val_split < 1, \
                (f'when {val_split=} is indicated as a float, it represents the fraction of training instances '
                 f'to be used for validation, and must thus be in the range (0,1)')
            assert fit_classifier, (f'when {val_split=} is indicated as a float (the fraction of training instances '
                                    f'to be used for validation), the parameter {fit_classifier=} must be True')
        elif isinstance(val_split, tuple):
            assert len(val_split) == 2, \
                (f'when {val_split=} is indicated as a tuple, it represents the collection (X,y) on which the '
                 f'validation must be performed, but this seems to have different cardinality')
        elif val_split is None:
            pass
        else:
            raise ValueError(f'unexpected type for {val_split=}')

        # classifier is fitted?
        try:
            check_is_fitted(self.classifier)
            fitted = True
        except NotFittedError:
            fitted = False

        # consistency checks: fit_classifier?
        if self.fit_classifier:
            if fitted:
                raise RuntimeWarning(f'the classifier is already fitted, by {fit_classifier=} was requested')
        else:
            assert fitted, (f'{fit_classifier=} requires the classifier to be already trained, '
                            f'but this does not seem to be')

    def _check_init_parameters(self):
        """
        Implements any check to be performed in the parameters of the init method before undertaking
        the training of the quantifier. This is made as to allow for a quick execution stop when the
        parameters are not valid.

        :return: Nothing. May raise an exception.
        """
        pass

    def _check_non_empty_classes(self, y):
        """
        Asserts all classes have positive instances.

        :param labels: array-like of shape `(n_instances,)` with the label for each instance
        :param classes: the class labels. This is needed in order to correctly compute the prevalence vector even when
            some classes have no examples.
        :return: Nothing. May raise an exception.
        """
        sample_prevs = F.prevalence_from_labels(y, self.classes_)
        empty_classes = np.argwhere(sample_prevs == 0).flatten()
        if len(empty_classes) > 0:
            empty_class_names = self.classes_[empty_classes]
            raise ValueError(f'classes {empty_class_names} have no training examples')

    def fit(self, X, y):
        """
        Trains the aggregative quantifier. This comes down to training a classifier (if requested) and an
        aggregation function.

        :param X: array-like of shape `(n_samples, n_features)`, the training instances
        :param y: array-like of shape `(n_samples,)`, the labels
        :return: self
        """
        self._check_init_parameters()
        classif_predictions, labels = self.classifier_fit_predict(X, y)
        self.aggregation_fit(classif_predictions, labels)
        return self

    def classifier_fit_predict(self, X, y):
        """
        Trains the classifier if requested (`fit_classifier=True`) and generate the necessary predictions to
        train the aggregation function.

        :param X: array-like of shape `(n_samples, n_features)`, the training instances
        :param y: array-like of shape `(n_samples,)`, the labels
        """
        self._check_classifier()

        # self._check_non_empty_classes(y)

        predictions, labels = None, None
        if isinstance(self.val_split, int):
            assert self.fit_classifier, f'unexpected value for {self.fit_classifier=}'
            num_folds = self.val_split
            n_jobs = self.n_jobs if hasattr(self, 'n_jobs') else qp._get_njobs(None)
            predictions = cross_val_predict(self.classifier, X, y, cv=num_folds, n_jobs=n_jobs, method=self._classifier_method())
            labels = y
            self.classifier.fit(X, y)
        elif isinstance(self.val_split, float):
            assert self.fit_classifier, f'unexpected value for {self.fit_classifier=}'
            train_prop = 1. - self.val_split
            Xtr, Xval, ytr, yval = train_test_split(X, y, train_size=train_prop, stratify=y)
            self.classifier.fit(Xtr, ytr)
            predictions = self.classify(Xval)
            labels = yval
        elif isinstance(self.val_split, tuple):
            Xval, yval = self.val_split
            if self.fit_classifier:
                self.classifier.fit(X, y)
            predictions = self.classify(Xval)
            labels = yval
        elif self.val_split is None:
            if self.fit_classifier:
                self.classifier.fit(X, y)
                predictions, labels = None, None
            else:
                predictions, labels = self.classify(X), y
        else:
            raise ValueError(f'unexpected type for {self.val_split=}')

        return predictions, labels

    @abstractmethod
    def aggregation_fit(self, classif_predictions, labels):
        """
        Trains the aggregation function.

        :param classif_predictions: array-like with  the classification predictions
            (whatever the method :meth:`classify` returns)
        :param labels: array-like with the true labels associated to each classifier prediction
        """
        ...

    @property
    def classifier(self):
        """
        Gives access to the classifier

        :return: the classifier (typically an sklearn's Estimator)
        """
        return self.classifier_

    @classifier.setter
    def classifier(self, classifier):
        """
        Setter for the classifier

        :param classifier: the classifier
        """
        self.classifier_ = classifier

    def classify(self, X):
        """
        Provides the label predictions for the given instances. The predictions should respect the format expected by
        :meth:`aggregate`, e.g., posterior probabilities for probabilistic quantifiers, or crisp predictions for
        non-probabilistic quantifiers. The default one is "decision_function".

        :param X: array-like of shape `(n_samples, n_features)`, the data instances
        :return: np.ndarray of shape `(n_instances,)` with classifier predictions
        """
        return getattr(self.classifier, self._classifier_method())(X)

    def _classifier_method(self):
        """
        Name of the method that must be used for issuing label predictions. The default one is "decision_function".

        :return: string
        """
        return 'decision_function'

    def _check_classifier(self, adapt_if_necessary=False):
        """
        Guarantees that the underlying classifier implements the method required for issuing predictions, i.e.,
        the method indicated by the :meth:`_classifier_method`

        :param adapt_if_necessary: unused unless overridden
        """
        assert hasattr(self.classifier, self._classifier_method()), \
            f"the method does not implement the required {self._classifier_method()} method"

    def predict(self, X):
        """
        Generate class prevalence estimates for the sample's instances by aggregating the label predictions generated
        by the classifier.

        :param X: array-like of shape `(n_samples, n_features)`, the data instances
        :return: `np.ndarray` of shape `(n_classes)` with class prevalence estimates.
        """
        classif_predictions = self.classify(X)
        return self.aggregate(classif_predictions)

    @abstractmethod
    def aggregate(self, classif_predictions: np.ndarray):
        """
        Implements the aggregation of the classifier predictions.

        :param classif_predictions: `np.ndarray` of classifier predictions
        :return: `np.ndarray` of shape `(n_classes,)` with class prevalence estimates.
        """
        ...

    @property
    def classes_(self):
        """
        Class labels, in the same order in which class prevalence values are to be computed.
        This default implementation actually returns the class labels of the learner.

        :return: array-like, the class labels
        """
        return self.classifier.classes_


class AggregativeCrispQuantifier(AggregativeQuantifier, ABC):
    """
    Abstract class for quantification methods that base their estimations on the aggregation of crisp decisions
    as returned by a hard classifier. Aggregative crisp quantifiers thus extend Aggregative
    Quantifiers by implementing specifications about crisp predictions.
    """

    def _classifier_method(self):
        """
        Name of the method that must be used for issuing label predictions. For crisp quantifiers, the method
        is 'predict', that returns an array of shape `(n_instances,)` of label predictions.

        :return: the string "predict", i.e., the standard method name for scikit-learn hard predictions
        """
        return 'predict'


class AggregativeSoftQuantifier(AggregativeQuantifier, ABC):
    """
    Abstract class for quantification methods that base their estimations on the aggregation of posterior
    probabilities as returned by a probabilistic classifier.
    Aggregative soft quantifiers thus extend Aggregative Quantifiers by implementing specifications
    about soft predictions.
    """

    def _classifier_method(self):
        """
        Name of the method that must be used for issuing label predictions. For probabilistic quantifiers, the method
        is 'predict_proba', that returns an array of shape `(n_instances, n_dimensions,)` with posterior
        probabilities.

        :return: the string "predict_proba", i.e., the standard method name for scikit-learn soft predictions
        """
        return 'predict_proba'

    def _check_classifier(self, adapt_if_necessary=False):
        """
        Guarantees that the underlying classifier implements the method indicated by the :meth:`_classifier_method`.
        In case it does not, the classifier is calibrated (by means of the Platt's calibration method implemented by
        scikit-learn in CalibratedClassifierCV, with cv=5). This calibration is only allowed if `adapt_if_necessary`
        is set to True. If otherwise (i.e., the classifier is not probabilistic, and `adapt_if_necessary` is set
        to False), an exception will be raised.

        :param adapt_if_necessary: a hard classifier is turned into a soft classifier if `adapt_if_necessary==True`
        """
        if not hasattr(self.classifier, self._classifier_method()):
            if adapt_if_necessary:
                print(f'warning: The learner {self.classifier.__class__.__name__} does not seem to be '
                      f'probabilistic. The learner will be calibrated (using CalibratedClassifierCV).')
                self.classifier = CalibratedClassifierCV(self.classifier, cv=5)
            else:
                raise AssertionError(f'error: The learner {self.classifier.__class__.__name__} does not '
                                     f'seem to be probabilistic. The learner cannot be calibrated since '
                                     f'fit_classifier is set to False')


class BinaryAggregativeQuantifier(AggregativeQuantifier, BinaryQuantifier):

    @property
    def pos_label(self):
        return self.classifier.classes_[1]

    @property
    def neg_label(self):
        return self.classifier.classes_[0]

    def fit(self, X, y):
        self._check_binary(y, self.__class__.__name__)
        return super().fit(X, y)


# Methods
# ------------------------------------
class CC(AggregativeCrispQuantifier):
    """
    The most basic Quantification method. One that simply classifies all instances and counts how many have been
    attributed to each of the classes in order to compute class prevalence estimates.

    :param classifier: a sklearn's Estimator that generates a classifier
    """
    def __init__(self, classifier: BaseEstimator = None, fit_classifier: bool = True):
        super().__init__(classifier, fit_classifier, val_split=None)

    def aggregation_fit(self, classif_predictions, labels):
        """
        Nothing to do here!

        :param classif_predictions: unused
        :param labels: unused
        """
        pass

    def aggregate(self, classif_predictions: np.ndarray):
        """
        Computes class prevalence estimates by counting the prevalence of each of the predicted labels.

        :param classif_predictions: array-like with classifier predictions
        :return: `np.ndarray` of shape `(n_classes,)` with class prevalence estimates.
        """
        return F.prevalence_from_labels(classif_predictions, self.classes_)


class PCC(AggregativeSoftQuantifier):
    """
    `Probabilistic Classify & Count <https://ieeexplore.ieee.org/abstract/document/5694031>`_,
    the probabilistic variant of CC that relies on the posterior probabilities returned by a probabilistic classifier.

    :param classifier: a sklearn's Estimator that generates a classifier
    """

    def __init__(self, classifier: BaseEstimator = None, fit_classifier: bool = True):
        super().__init__(classifier, fit_classifier, val_split=None)

    def aggregation_fit(self, classif_predictions, labels):
        """
        Nothing to do here!

        :param classif_predictions: unused
        :param labels: unused
        """
        pass

    def aggregate(self, classif_posteriors):
        return F.prevalence_from_probabilities(classif_posteriors, binarize=False)


class ACC(AggregativeCrispQuantifier):
    """
    `Adjusted Classify & Count <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_,
    the "adjusted" variant of :class:`CC`, that corrects the predictions of CC
    according to the `misclassification rates`.

    :param classifier: a scikit-learn's BaseEstimator, or None, in which case the classifier is taken to be
        the one indicated in `qp.environ['DEFAULT_CLS']`

    :param fit_classifier: whether to train the learner (default is True). Set to False if the
        learner has been trained outside the quantifier.

    :param val_split: specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer (default 5), indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`); or as a tuple (X,y) defining the specific set of data to use for validation.

    :param str method: adjustment method to be used:

        * 'inversion': matrix inversion method based on the matrix equality :math:`P(C)=P(C|Y)P(Y)`,
          which tries to invert :math:`P(C|Y)` matrix.
        * 'invariant-ratio': invariant ratio estimator of `Vaz et al. 2018 <https://jmlr.org/papers/v20/18-456.html>`_,
          which replaces the last equation with the normalization condition.

    :param str solver: indicates the method to use for solving the system of linear equations. Valid options are:

        * 'exact-raise': tries to solve the system using matrix inversion. Raises an error if the matrix has rank
          strictly less than `n_classes`.
        * 'exact-cc': if the matrix is not of full rank, returns `p_c` as the estimates, which corresponds to
          no adjustment (i.e., the classify and count method. See :class:`quapy.method.aggregative.CC`)
        * 'exact': deprecated, defaults to 'exact-cc'
        * 'minimize': minimizes the L2 norm of :math:`|Ax-B|`. This one generally works better, and is the
          default parameter. More details about this can be consulted in `Bunse, M. "On Multi-Class Extensions of
          Adjusted Classify and Count", on proceedings of the 2nd International Workshop on Learning to Quantify:
          Methods and Applications (LQ 2022), ECML/PKDD 2022, Grenoble (France)
          <https://lq-2022.github.io/proceedings/CompleteVolume.pdf>`_.

    :param str norm: the method to use for normalization.

        * `clip`, the values are clipped to the range [0,1] and then L1-normalized.
        * `mapsimplex` projects vectors onto the probability simplex. This implementation relies on
          `Mathieu Blondel's projection_simplex_sort <https://gist.github.com/mblondel/6f3b7aaad90606b98f71>`_
        * `condsoftmax`, applies a softmax normalization only to prevalence vectors that lie outside the simplex

    :param n_jobs: number of parallel workers
    """

    def __init__(
            self,
            classifier: BaseEstimator = None,
            fit_classifier = True,
            val_split = 5,
            solver: Literal['minimize', 'exact', 'exact-raise', 'exact-cc'] = 'minimize',
            method: Literal['inversion', 'invariant-ratio'] = 'inversion',
            norm: Literal['clip', 'mapsimplex', 'condsoftmax'] = 'clip',
            n_jobs=None,
    ):
        super().__init__(classifier, fit_classifier, val_split)
        self.n_jobs = qp._get_njobs(n_jobs)
        self.solver = solver
        self.method = method
        self.norm = norm

    SOLVERS = ['exact', 'minimize', 'exact-raise', 'exact-cc']
    METHODS = ['inversion', 'invariant-ratio']
    NORMALIZATIONS = ['clip', 'mapsimplex', 'condsoftmax', None]

    @classmethod
    def newInvariantRatioEstimation(cls, classifier: BaseEstimator, fit_classifier=True, val_split=5, n_jobs=None):
        """
        Constructs a quantifier that implements the Invariant Ratio Estimator of
        `Vaz et al. 2018 <https://jmlr.org/papers/v20/18-456.html>`_. This amounts
        to setting method to 'invariant-ratio' and clipping to 'project'.

        :param classifier: a scikit-learn's BaseEstimator, or None, in which case the classifier is taken to be
            the one indicated in `qp.environ['DEFAULT_CLS']`

        :param fit_classifier: whether to train the learner (default is True). Set to False if the
            learner has been trained outside the quantifier.

        :param val_split: specifies the data used for generating classifier predictions. This specification
            can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
            be extracted from the training set; or as an integer (default 5), indicating that the predictions
            are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
            for `k`); or as a tuple (X,y) defining the specific set of data to use for validation.

        :param n_jobs: number of parallel workers

        :return: an instance of ACC configured so that it implements the Invariant Ratio Estimator
        """
        return ACC(classifier, fit_classifier=fit_classifier, val_split=val_split, method='invariant-ratio', norm='mapsimplex', n_jobs=n_jobs)

    def _check_init_parameters(self):
        if self.solver not in ACC.SOLVERS:
            raise ValueError(f"unknown solver; valid ones are {ACC.SOLVERS}")
        if self.method not in ACC.METHODS:
            raise ValueError(f"unknown method; valid ones are {ACC.METHODS}")
        if self.norm not in ACC.NORMALIZATIONS:
            raise ValueError(f"unknown normalization; valid ones are {ACC.NORMALIZATIONS}")

    def aggregation_fit(self, classif_predictions, labels):
        """
        Estimates the misclassification rates.
        :param classif_predictions: array-like with the predicted labels
        :param labels: array-like with the true labels associated to each predicted label
        """
        true_labels = labels
        pred_labels = classif_predictions
        self.cc = CC(self.classifier, fit_classifier=False)
        self.Pte_cond_estim_ = ACC.getPteCondEstim(self.classifier.classes_, true_labels, pred_labels)

    @classmethod
    def getPteCondEstim(cls, classes, y, y_):
        """
        Estimate the matrix with entry (i,j) being the estimate of P(hat_yi|yj), that is, the probability that a
        document that belongs to yj ends up being classified as belonging to yi

        :param classes: array-like with the class names
        :param y: array-like with the true labels
        :param y_: array-like with the estimated labels
        :return: np.ndarray
        """
        conf = confusion_matrix(y, y_, labels=classes).T
        conf = conf.astype(float)
        class_counts = conf.sum(axis=0)
        for i, _ in enumerate(classes):
            if class_counts[i] == 0:
                conf[i, i] = 1
            else:
                conf[:, i] /= class_counts[i]
        return conf

    def aggregate(self, classif_predictions):
        prevs_estim = self.cc.aggregate(classif_predictions)
        estimate = F.solve_adjustment(
            class_conditional_rates=self.Pte_cond_estim_,
            unadjusted_counts=prevs_estim,
            solver=self.solver,
            method=self.method,
        )
        return F.normalize_prevalence(estimate, method=self.norm)


class PACC(AggregativeSoftQuantifier):
    """
    `Probabilistic Adjusted Classify & Count <https://ieeexplore.ieee.org/abstract/document/5694031>`_,
    the probabilistic variant of ACC that relies on the posterior probabilities returned by a probabilistic classifier.

    :param classifier: a scikit-learn's BaseEstimator, or None, in which case the classifier is taken to be
        the one indicated in `qp.environ['DEFAULT_CLS']`

    :param fit_classifier: whether to train the learner (default is True). Set to False if the
        learner has been trained outside the quantifier.

    :param val_split: specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer (default 5), indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`); or as a tuple (X,y) defining the specific set of data to use for validation.

    :param str method: adjustment method to be used:

        * 'inversion': matrix inversion method based on the matrix equality :math:`P(C)=P(C|Y)P(Y)`,
          which tries to invert `P(C|Y)` matrix.
        * 'invariant-ratio': invariant ratio estimator of `Vaz et al. <https://jmlr.org/papers/v20/18-456.html>`_,
          which replaces the last equation with the normalization condition.

    :param str solver: the method to use for solving the system of linear equations. Valid options are:

        * 'exact-raise': tries to solve the system using matrix inversion.
          Raises an error if the matrix has rank strictly less than `n_classes`.
        * 'exact-cc': if the matrix is not of full rank, returns `p_c` as the estimates, which
          corresponds to no adjustment (i.e., the classify and count method. See :class:`quapy.method.aggregative.CC`)
        * 'exact': deprecated, defaults to 'exact-cc'
        * 'minimize': minimizes the L2 norm of :math:`|Ax-B|`. This one generally works better, and is the
          default parameter. More details about this can be consulted in `Bunse, M. "On Multi-Class Extensions
          of Adjusted Classify and Count", on proceedings of the 2nd International Workshop on Learning to
          Quantify: Methods and Applications (LQ 2022), ECML/PKDD 2022, Grenoble (France)
          <https://lq-2022.github.io/proceedings/CompleteVolume.pdf>`_.

    :param str norm: the method to use for normalization.

        * `clip`, the values are clipped to the range [0,1] and then L1-normalized.
        * `mapsimplex` projects vectors onto the probability simplex. This implementation relies on
          `Mathieu Blondel's projection_simplex_sort <https://gist.github.com/mblondel/6f3b7aaad90606b98f71>`_
        * `condsoftmax`, applies a softmax normalization only to prevalence vectors that lie outside the simplex

    :param n_jobs: number of parallel workers
    """

    def __init__(
            self,
            classifier: BaseEstimator = None,
            fit_classifier=True,
            val_split=5,
            solver: Literal['minimize', 'exact', 'exact-raise', 'exact-cc'] = 'minimize',
            method: Literal['inversion', 'invariant-ratio'] = 'inversion',
            norm: Literal['clip', 'mapsimplex', 'condsoftmax'] = 'clip',
            n_jobs=None
    ):
        super().__init__(classifier, fit_classifier, val_split)
        self.n_jobs = qp._get_njobs(n_jobs)
        self.solver = solver
        self.method = method
        self.norm = norm

    def _check_init_parameters(self):
        if self.solver not in ACC.SOLVERS:
            raise ValueError(f"unknown solver; valid ones are {ACC.SOLVERS}")
        if self.method not in ACC.METHODS:
            raise ValueError(f"unknown method; valid ones are {ACC.METHODS}")
        if self.norm not in ACC.NORMALIZATIONS:
            raise ValueError(f"unknown normalization; valid ones are {ACC.NORMALIZATIONS}")

    def aggregation_fit(self, classif_predictions, labels):
        """
        Estimates the misclassification rates

        :param classif_predictions: array-like with posterior probabilities
        :param labels: array-like with the true labels associated to each vector of posterior probabilities
        """
        posteriors = classif_predictions
        true_labels = labels
        self.pcc = PCC(self.classifier, fit_classifier=False)
        self.Pte_cond_estim_ = PACC.getPteCondEstim(self.classifier.classes_, true_labels, posteriors)

    def aggregate(self, classif_posteriors):
        prevs_estim = self.pcc.aggregate(classif_posteriors)

        estimate = F.solve_adjustment(
            class_conditional_rates=self.Pte_cond_estim_,
            unadjusted_counts=prevs_estim,
            solver=self.solver,
            method=self.method,
        )
        return F.normalize_prevalence(estimate, method=self.norm)

    @classmethod
    def getPteCondEstim(cls, classes, y, y_):
        # estimate the matrix with entry (i,j) being the estimate of P(hat_yi|yj), that is, the probability that a
        # document that belongs to yj ends up being classified as belonging to yi
        n_classes = len(classes)
        confusion = np.eye(n_classes)
        for i, class_ in enumerate(classes):
            idx = y == class_
            if idx.any():
                confusion[i] = y_[idx].mean(axis=0)

        return confusion.T


class EMQ(AggregativeSoftQuantifier):
    """
    `Expectation Maximization for Quantification <https://ieeexplore.ieee.org/abstract/document/6789744>`_ (EMQ),
    aka `Saerens-Latinne-Decaestecker` (SLD) algorithm.
    EMQ consists of using the well-known `Expectation Maximization algorithm` to iteratively update the posterior
    probabilities generated by a probabilistic classifier and the class prevalence estimates obtained via
    maximum-likelihood estimation, in a mutually recursive way, until convergence.

    This implementation also gives access to the heuristics proposed by `Alexandari et al. paper
    <http://proceedings.mlr.press/v119/alexandari20a.html>`_. These heuristics consist of using, as the training
    prevalence, an estimate of it obtained via k-fold cross validation (instead of the true training prevalence),
    and to recalibrate the posterior probabilities of the classifier.

    :param classifier: a scikit-learn's BaseEstimator, or None, in which case the classifier is taken to be
        the one indicated in `qp.environ['DEFAULT_CLS']`

    :param fit_classifier: whether to train the learner (default is True). Set to False if the
        learner has been trained outside the quantifier.

    :param val_split: specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer (default 5), indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`); or as a tuple (X,y) defining the specific set of data to use for validation.
        This hyperparameter is only meant to be used when the heuristics are to be applied, i.e., if a
        calibration is required. The default value is None (meaning the calibration is not required). In
        case this hyperparameter is set to a value other than None, but the calibration is not required
        (calib=None), a warning message will be raised.

    :param exact_train_prev: set to True (default) for using the true training prevalence as the initial observation;
        set to False for computing the training prevalence as an estimate of it, i.e., as the expected
        value of the posterior probabilities of the training instances.

    :param calib: a string indicating the method of calibration.
        Available choices include "nbvs" (No-Bias Vector Scaling), "bcts" (Bias-Corrected Temperature Scaling,
        default), "ts" (Temperature Scaling), and "vs" (Vector Scaling). Default is None (no calibration).

    :param on_calib_error: a string indicating the policy to follow in case the calibrator fails at runtime.
        Options include "raise" (default), in which case a RuntimeException is raised; and "backup", in which
        case the calibrator is silently skipped.

    :param n_jobs: number of parallel workers. Only used for recalibrating the classifier if `val_split` is set to
        an integer `k` --the number of folds.
    """

    MAX_ITER = 1000
    EPSILON = 1e-4
    ON_CALIB_ERROR_VALUES = ['raise', 'backup']
    CALIB_OPTIONS = [None, 'nbvs', 'bcts', 'ts', 'vs']

    def __init__(self, classifier: BaseEstimator = None, fit_classifier=True, val_split=None, exact_train_prev=True,
                 calib=None, on_calib_error='raise', n_jobs=None):

        assert calib in EMQ.CALIB_OPTIONS, \
            f'invalid value for {calib=}; valid ones are {EMQ.CALIB_OPTIONS}'
        assert on_calib_error in EMQ.ON_CALIB_ERROR_VALUES, \
            f'invalid value for {on_calib_error=}; valid ones are {EMQ.ON_CALIB_ERROR_VALUES}'

        super().__init__(classifier, fit_classifier, val_split)
        self.exact_train_prev = exact_train_prev
        self.calib = calib
        self.on_calib_error = on_calib_error
        self.n_jobs = n_jobs

    @classmethod
    def EMQ_BCTS(cls, classifier: BaseEstimator, fit_classifier=True, val_split=5, on_calib_error="raise", n_jobs=None):
        """
        Constructs an instance of EMQ using the best configuration found in the `Alexandari et al. paper
        <http://proceedings.mlr.press/v119/alexandari20a.html>`_, i.e., one that relies on Bias-Corrected Temperature
        Scaling (BCTS) as a calibration function, and that uses an estimate of the training prevalence instead of
        the true training prevalence.

        :param classifier: a scikit-learn's BaseEstimator, or None, in which case the classifier is taken to be
            the one indicated in `qp.environ['DEFAULT_CLS']`

        :param fit_classifier: whether to train the learner (default is True). Set to False if the
            learner has been trained outside the quantifier.

        :param val_split: specifies the data used for generating classifier predictions. This specification
            can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
            be extracted from the training set; or as an integer (default 5), indicating that the predictions
            are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
            for `k`); or as a tuple (X,y) defining the specific set of data to use for validation.

        :param on_calib_error: a string indicating the policy to follow in case the calibrator fails at runtime.
            Options include "raise" (default), in which case a RuntimeException is raised; and "backup", in which
            case the calibrator is silently skipped.

        :param n_jobs: number of parallel workers. Only used for recalibrating the classifier if `val_split` is set to
            an integer `k` --the number of folds.

        :return: An instance of EMQ with BCTS
        """
        return EMQ(classifier, fit_classifier=fit_classifier, val_split=val_split, exact_train_prev=False,
                   calib='bcts', on_calib_error=on_calib_error, n_jobs=n_jobs)

    def _check_init_parameters(self):
        if self.val_split is not None:
            if self.exact_train_prev and self.calib is None:
                raise RuntimeWarning(f'The parameter {self.val_split=} was specified for EMQ, while the parameters '
                                     f'{self.exact_train_prev=} and {self.calib=}. This has no effect and causes an unnecessary '
                                     f'overload.')
        else:
            if self.calib is not None:
                print(f'[warning] The parameter {self.calib=} requires the val_split be different from None. '
                      f'This parameter will be set to 5. To avoid this warning, set this value to a float value '
                      f'indicating the proportion of training data to be used as validation, or to an integer '
                      f'indicating the number of folds for kFCV.')
                self.val_split = 5

    def classify(self, X):
        """
        Provides the posterior probabilities for the given instances. The calibration function, if required,
        has no effect in this step, and is only involved in the aggregate method.

        :param X: array-like of shape `(n_instances, n_dimensions,)`
        :return: np.ndarray of shape `(n_instances, n_classes,)` with posterior probabilities
        """
        return self.classifier.predict_proba(X)

    def classifier_fit_predict(self, X, y):
        classif_predictions = super().classifier_fit_predict(X, y)
        self.train_prevalence = F.prevalence_from_labels(y, classes=self.classes_)
        return classif_predictions

    def _fit_calibration(self, calibrator, P, y):
        n_classes = len(self.classes_)

        if not np.issubdtype(y.dtype, np.number):
            y = np.searchsorted(self.classes_, y)

        try:
            self.calibration_function = calibrator(P, np.eye(n_classes)[y], posterior_supplied=True)
        except Exception as e:
            if self.on_calib_error == 'raise':
                raise RuntimeError(f'calibration {self.calib} failed at fit time: {e}')
            elif self.on_calib_error == 'backup':
                self.calibration_function = lambda P: P

    def _calibrate_if_requested(self, uncalib_posteriors):
        if hasattr(self, 'calibration_function') and self.calibration_function is not None:
            try:
                calib_posteriors = self.calibration_function(uncalib_posteriors)
            except Exception as e:
                if self.on_calib_error == 'raise':
                    raise RuntimeError(f'calibration {self.calib} failed at predict time: {e}')
                elif self.on_calib_error == 'backup':
                    calib_posteriors = uncalib_posteriors
                else:
                    raise ValueError(f'unexpected {self.on_calib_error=}; '
                                     f'valid options are {EMQ.ON_CALIB_ERROR_VALUES}')
            return calib_posteriors
        return uncalib_posteriors

    def aggregation_fit(self, classif_predictions, labels):
        """
        Trains the aggregation function of EMQ. This comes down to recalibrating the posterior probabilities
        ir requested.

        :param classif_predictions: array-like with the raw (i.e., uncalibrated) posterior probabilities
            returned by the classifier
        :param labels: array-like with the true labels associated to each classifier prediction
        """
        P = classif_predictions
        y = labels
        if self.calib is not None:
            calibrator = {
                'nbvs': NoBiasVectorScaling(),
                'bcts': TempScaling(bias_positions='all'),
                'ts': TempScaling(),
                'vs': VectorScaling()
            }.get(self.calib, None)

            if calibrator is None:
                raise ValueError(f'invalid value for {self.calib=}; valid ones are {EMQ.CALIB_OPTIONS}')

            self._fit_calibration(calibrator, P, y)

        if not self.exact_train_prev:
            P = self._calibrate_if_requested(P)
            self.train_prevalence = F.prevalence_from_probabilities(P)

    def aggregate(self, classif_posteriors, epsilon=EPSILON):
        classif_posteriors = self._calibrate_if_requested(classif_posteriors)
        priors, posteriors = self.EM(self.train_prevalence, classif_posteriors, epsilon)
        return priors

    def predict_proba(self, instances, epsilon=EPSILON):
        """
        Returns the posterior probabilities updated by the EM algorithm.

        :param instances: np.ndarray of shape `(n_instances, n_dimensions)`
        :param epsilon: error tolerance
        :return: np.ndarray of shape `(n_instances, n_classes)`
        """
        classif_posteriors = self.classify(instances)
        classif_posteriors = self._calibrate_if_requested(classif_posteriors)
        priors, posteriors = self.EM(self.train_prevalence, classif_posteriors, epsilon)
        return posteriors

    @classmethod
    def EM(cls, tr_prev, posterior_probabilities, epsilon=EPSILON):
        """
        Computes the `Expectation Maximization` routine.

        :param tr_prev: array-like, the training prevalence
        :param posterior_probabilities: `np.ndarray` of shape `(n_instances, n_classes,)` with the
            posterior probabilities
        :param epsilon: float, the threshold different between two consecutive iterations
            to reach before stopping the loop
        :return: a tuple with the estimated prevalence values (shape `(n_classes,)`) and
            the corrected posterior probabilities (shape `(n_instances, n_classes,)`)
        """
        Px = posterior_probabilities
        Ptr = np.copy(tr_prev)

        if np.prod(Ptr) == 0:  # some entry is 0; we should smooth the values to avoid 0 division
            Ptr += epsilon
            Ptr /= Ptr.sum()

        qs = np.copy(Ptr)  # qs (the running estimate) is initialized as the training prevalence

        s, converged = 0, False
        qs_prev_ = None
        while not converged and s < EMQ.MAX_ITER:
            # E-step: ps is Ps(y|xi)
            ps_unnormalized = (qs / Ptr) * Px
            ps = ps_unnormalized / ps_unnormalized.sum(axis=1, keepdims=True)

            # M-step:
            qs = ps.mean(axis=0)

            if qs_prev_ is not None and qp.error.mae(qs, qs_prev_) < epsilon and s > 10:
                converged = True

            qs_prev_ = qs
            s += 1

        if not converged:
            print('[warning] the method has reached the maximum number of iterations; it might have not converged')

        return qs, ps


class HDy(AggregativeSoftQuantifier, BinaryAggregativeQuantifier):
    """
    `Hellinger Distance y <https://www.sciencedirect.com/science/article/pii/S0020025512004069>`_ (HDy).
    HDy is a probabilistic method for training binary quantifiers, that models quantification as the problem of
    minimizing the divergence (in terms of the Hellinger Distance) between two distributions of posterior
    probabilities returned by the classifier. One of the distributions is generated from the unlabelled examples and
    the other is generated from a validation set. This latter distribution is defined as a mixture of the
    class-conditional distributions of the posterior probabilities returned for the positive and negative validation
    examples, respectively. The parameters of the mixture thus represent the estimates of the class prevalence values.

    :param classifier: a scikit-learn's BaseEstimator, or None, in which case the classifier is taken to be
        the one indicated in `qp.environ['DEFAULT_CLS']`

    :param fit_classifier: whether to train the learner (default is True). Set to False if the
        learner has been trained outside the quantifier.

    :param val_split: specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer (default 5), indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`); or as a tuple (X,y) defining the specific set of data to use for validation.
    """

    def __init__(self, classifier: BaseEstimator = None, fit_classifier=True, val_split=5):
        super().__init__(classifier, fit_classifier, val_split)

    def aggregation_fit(self, classif_predictions, labels):
        """
        Trains the aggregation function of HDy.

        :param classif_predictions: array-like with the posterior probabilities returned by the classifier
        :param labels: array-like with the true labels associated to each posterior
        """
        P, y = classif_predictions, labels
        Px = P[:, self.pos_label]  # takes only the P(y=+1|x)
        self.Pxy1 = Px[y == self.pos_label]
        self.Pxy0 = Px[y == self.neg_label]

        # pre-compute the histogram for positive and negative examples
        self.bins = np.linspace(10, 110, 11, dtype=int)  # [10, 20, 30, ..., 100, 110]

        def hist(P, bins):
            h = np.histogram(P, bins=bins, range=(0, 1), density=True)[0]
            return h / h.sum()

        self.Pxy1_density = {bins: hist(self.Pxy1, bins) for bins in self.bins}
        self.Pxy0_density = {bins: hist(self.Pxy0, bins) for bins in self.bins}

    def aggregate(self, classif_posteriors):
        # "In this work, the number of bins b used in HDx and HDy was chosen from 10 to 110 in steps of 10,
        # and the final estimated a priori probability was taken as the median of these 11 estimates."
        # (González-Castro, et al., 2013).

        Px = classif_posteriors[:, self.pos_label]  # takes only the P(y=+1|x)

        prev_estimations = []
        # for bins in np.linspace(10, 110, 11, dtype=int):  #[10, 20, 30, ..., 100, 110]
        # Pxy0_density, _ = np.histogram(self.Pxy0, bins=bins, range=(0, 1), density=True)
        # Pxy1_density, _ = np.histogram(self.Pxy1, bins=bins, range=(0, 1), density=True)
        for bins in self.bins:
            Pxy0_density = self.Pxy0_density[bins]
            Pxy1_density = self.Pxy1_density[bins]

            Px_test, _ = np.histogram(Px, bins=bins, range=(0, 1), density=True)

            # the authors proposed to search for the prevalence yielding the best matching as a linear search
            # at small steps (modern implementations resort to an optimization procedure,
            # see class DistributionMatching)
            prev_selected, min_dist = None, None
            for prev in F.prevalence_linspace(grid_points=101, repeats=1, smooth_limits_epsilon=0.0):
                Px_train = prev * Pxy1_density + (1 - prev) * Pxy0_density
                hdy = F.HellingerDistance(Px_train, Px_test)
                if prev_selected is None or hdy < min_dist:
                    prev_selected, min_dist = prev, hdy
            prev_estimations.append(prev_selected)

        class1_prev = np.median(prev_estimations)
        return F.as_binary_prevalence(class1_prev)


class DyS(AggregativeSoftQuantifier, BinaryAggregativeQuantifier):
    """
    `DyS framework <https://ojs.aaai.org/index.php/AAAI/article/view/4376>`_ (DyS).
    DyS is a generalization of HDy method, using a Ternary Search in order to find the prevalence that
    minimizes the distance between distributions.
    Details for the ternary search have been got from <https://dl.acm.org/doi/pdf/10.1145/3219819.3220059>

    :param classifier: a scikit-learn's BaseEstimator, or None, in which case the classifier is taken to be
        the one indicated in `qp.environ['DEFAULT_CLS']`

    :param fit_classifier: whether to train the learner (default is True). Set to False if the
        learner has been trained outside the quantifier.

    :param val_split: specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer (default 5), indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`); or as a tuple (X,y) defining the specific set of data to use for validation.

    :param n_bins: an int with the number of bins to use to compute the histograms.

    :param divergence: a str indicating the name of divergence (currently supported ones are "HD" or "topsoe"), or a
        callable function computes the divergence between two distributions (two equally sized arrays).

    :param tol: a float with the tolerance for the ternary search algorithm.

    :param n_jobs: number of parallel workers.
    """

    def __init__(self, classifier: BaseEstimator = None, fit_classifier=True, val_split=5, n_bins=8,
                 divergence: Union[str, Callable] = 'HD', tol=1e-05, n_jobs=None):
        super().__init__(classifier, fit_classifier, val_split)
        self.tol = tol
        self.divergence = divergence
        self.n_bins = n_bins
        self.n_jobs = n_jobs

    def _ternary_search(self, f, left, right, tol):
        """
        Find maximum of unimodal function f() within [left, right]
        """
        while abs(right - left) >= tol:
            left_third = left + (right - left) / 3
            right_third = right - (right - left) / 3

            if f(left_third) > f(right_third):
                left = left_third
            else:
                right = right_third

        # Left and right are the current bounds; the maximum is between them
        return (left + right) / 2

    def aggregation_fit(self, classif_predictions, labels):
        """
        Trains the aggregation function of DyS.

        :param classif_predictions: array-like with the posterior probabilities returned by the classifier
        :param labels: array-like with the true labels associated to each posterior
        """
        Px, y = classif_predictions, labels
        Px = Px[:, self.pos_label]  # takes only the P(y=+1|x)
        self.Pxy1 = Px[y == self.pos_label]
        self.Pxy0 = Px[y == self.neg_label]
        self.Pxy1_density = np.histogram(self.Pxy1, bins=self.n_bins, range=(0, 1), density=True)[0]
        self.Pxy0_density = np.histogram(self.Pxy0, bins=self.n_bins, range=(0, 1), density=True)[0]
        return self

    def aggregate(self, classif_posteriors):
        Px = classif_posteriors[:, self.pos_label]  # takes only the P(y=+1|x)

        Px_test = np.histogram(Px, bins=self.n_bins, range=(0, 1), density=True)[0]
        divergence = get_divergence(self.divergence)

        def distribution_distance(prev):
            Px_train = prev * self.Pxy1_density + (1 - prev) * self.Pxy0_density
            return divergence(Px_train, Px_test)

        class1_prev = self._ternary_search(f=distribution_distance, left=0, right=1, tol=self.tol)
        return F.as_binary_prevalence(class1_prev)


class SMM(AggregativeSoftQuantifier, BinaryAggregativeQuantifier):
    """
    `SMM method <https://ieeexplore.ieee.org/document/9260028>`_ (SMM).
    SMM is a simplification of matching distribution methods where the representation of the examples
    is created using the mean instead of a histogram (conceptually equivalent to PACC).

    :param classifier: a scikit-learn's BaseEstimator, or None, in which case the classifier is taken to be
        the one indicated in `qp.environ['DEFAULT_CLS']`

    :param fit_classifier: whether to train the learner (default is True). Set to False if the
        learner has been trained outside the quantifier.

    :param val_split: specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer (default 5), indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`); or as a tuple (X,y) defining the specific set of data to use for validation.
    """

    def __init__(self, classifier: BaseEstimator = None, fit_classifier=True, val_split=5):
        super().__init__(classifier, fit_classifier, val_split)

    def aggregation_fit(self, classif_predictions, labels):
        """
        Trains the aggregation function of SMM.

        :param classif_predictions: array-like with the posterior probabilities returned by the classifier
        :param labels: array-like with the true labels associated to each posterior
        """
        Px, y = classif_predictions, labels
        Px = Px[:, self.pos_label]  # takes only the P(y=+1|x)
        self.Pxy1 = Px[y == self.pos_label]
        self.Pxy0 = Px[y == self.neg_label]
        self.Pxy1_mean = np.mean(self.Pxy1)  # equiv. TPR
        self.Pxy0_mean = np.mean(self.Pxy0)  # equiv. FPR
        return self

    def aggregate(self, classif_posteriors):
        Px = classif_posteriors[:, self.pos_label]  # takes only the P(y=+1|x)
        Px_mean = np.mean(Px)

        class1_prev = (Px_mean - self.Pxy0_mean) / (self.Pxy1_mean - self.Pxy0_mean)
        return F.as_binary_prevalence(class1_prev, clip_if_necessary=True)


class DMy(AggregativeSoftQuantifier):
    """
    Generic Distribution Matching quantifier for binary or multiclass quantification based on the space of posterior
    probabilities. This implementation takes the number of bins, the divergence, and the possibility to work on CDF
    as hyperparameters.

    :param classifier: a scikit-learn's BaseEstimator, or None, in which case the classifier is taken to be
        the one indicated in `qp.environ['DEFAULT_CLS']`

    :param fit_classifier: whether to train the learner (default is True). Set to False if the
        learner has been trained outside the quantifier.

    :param val_split: specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer (default 5), indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`); or as a tuple (X,y) defining the specific set of data to use for validation.

    :param nbins: number of bins used to discretize the distributions (default 8)

    :param divergence: a string representing a divergence measure (currently, "HD" and "topsoe" are implemented)
        or a callable function taking two ndarrays of the same dimension as input (default "HD", meaning Hellinger
        Distance)

    :param cdf: whether to use CDF instead of PDF (default False)

    :param n_jobs: number of parallel workers (default None)
    """

    def __init__(self, classifier: BaseEstimator = None, fit_classifier=True, val_split=5, nbins=8,
                 divergence: Union[str, Callable] = 'HD', cdf=False, search='optim_minimize', n_jobs=None):
        super().__init__(classifier, fit_classifier, val_split)
        self.nbins = nbins
        self.divergence = divergence
        self.cdf = cdf
        self.search = search
        self.n_jobs = n_jobs

    # @classmethod
    # def HDy(cls, classifier, val_split=5, n_jobs=None):
    #     from quapy.method.meta import MedianEstimator
    #
    #     hdy = DMy(classifier=classifier, val_split=val_split, search='linear_search', divergence='HD')
    #     hdy = AggregativeMedianEstimator(hdy, param_grid={'nbins': np.linspace(10, 110, 11).astype(int)}, n_jobs=n_jobs)
    #     return hdy

    def _get_distributions(self, posteriors):
        histograms = []
        post_dims = posteriors.shape[1]
        if post_dims == 2:
            # in binary quantification we can use only one class, since the other one is its complement
            post_dims = 1
        for dim in range(post_dims):
            hist = np.histogram(posteriors[:, dim], bins=self.nbins, range=(0, 1))[0]
            histograms.append(hist)

        counts = np.vstack(histograms)
        distributions = counts / counts.sum(axis=1)[:, np.newaxis]
        if self.cdf:
            distributions = np.cumsum(distributions, axis=1)
        return distributions

    def aggregation_fit(self, classif_predictions, labels):
        """
        Trains the aggregation function of a distribution matching method. This comes down to generating the
        validation distributions out of the training data.
        The validation distributions have shape `(n, ch, nbins)`, with `n` the number of classes, `ch` the number of
        channels, and `nbins` the number of bins. In particular, let `V` be the validation distributions; then `di=V[i]`
        are the distributions obtained from training data labelled with class `i`; while `dij = di[j]` is the discrete
        distribution of posterior probabilities `P(Y=j|X=x)` for training data labelled with class `i`, and `dij[k]`
        is the fraction of instances with a value in the `k`-th bin.

        :param classif_predictions: array-like with the posterior probabilities returned by the classifier
        :param labels: array-like with the true labels associated to each posterior
        """
        posteriors, true_labels = classif_predictions, labels
        n_classes = len(self.classifier.classes_)

        self.validation_distribution = qp.util.parallel(
            func=self._get_distributions,
            args=[posteriors[true_labels == cat] for cat in range(n_classes)],
            n_jobs=self.n_jobs,
            backend='threading'
        )

    def aggregate(self, posteriors: np.ndarray):
        """
        Searches for the mixture model parameter (the sought prevalence values) that yields a validation distribution
        (the mixture) that best matches the test distribution, in terms of the divergence measure of choice.
        In the multiclass case, with `n` the number of classes, the test and mixture distributions contain
        `n` channels (proper distributions of binned posterior probabilities), on which the divergence is computed
        independently. The matching is computed as an average of the divergence across all channels.

        :param posteriors: posterior probabilities of the instances in the sample
        :return: a vector of class prevalence estimates
        """
        test_distribution = self._get_distributions(posteriors)
        divergence = get_divergence(self.divergence)
        n_classes, n_channels, nbins = self.validation_distribution.shape

        def loss(prev):
            prev = np.expand_dims(prev, axis=0)
            mixture_distribution = (prev @ self.validation_distribution.reshape(n_classes, -1)).reshape(n_channels, -1)
            divs = [divergence(test_distribution[ch], mixture_distribution[ch]) for ch in range(n_channels)]
            return np.mean(divs)

        return F.argmin_prevalence(loss, n_classes, method=self.search)


def newELM(svmperf_base=None, loss='01', C=1):
    """
    Explicit Loss Minimization (ELM) quantifiers.
    Quantifiers based on ELM represent a family of methods based on structured output learning;
    these quantifiers rely on classifiers that have been optimized using a quantification-oriented loss
    measure. This implementation relies on
    `Joachims’ SVM perf <https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html>`_ structured output
    learning algorithm, which has to be installed and patched for the purpose (see this
    `script <https://github.com/HLT-ISTI/QuaPy/blob/master/prepare_svmperf.sh>`_).
    This function equivalent to:

    >>> CC(SVMperf(svmperf_base, loss, C))

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`; if set to None (default)
        this path will be obtained from qp.environ['SVMPERF_HOME']
    :param loss: the loss to optimize (see :attr:`quapy.classification.svmperf.SVMperf.valid_losses`)
    :param C: trade-off between training error and margin (default 0.01)
    :return: returns an instance of CC set to work with SVMperf (with loss and C set properly) as the
        underlying classifier
    """
    if svmperf_base is None:
        svmperf_base = qp.environ['SVMPERF_HOME']
    assert svmperf_base is not None, \
        'param svmperf_base was not specified, and the variable SVMPERF_HOME has not been set in the environment'
    return CC(SVMperf(svmperf_base, loss=loss, C=C))


def newSVMQ(svmperf_base=None, C=1):
    """
    SVM(Q) is an Explicit Loss Minimization (ELM) quantifier set to optimize for the `Q` loss combining a
    classification-oriented loss and a quantification-oriented loss, as proposed by
    `Barranquero et al. 2015 <https://www.sciencedirect.com/science/article/pii/S003132031400291X>`_.
    Equivalent to:

    >>> CC(SVMperf(svmperf_base, loss='q', C=C))

    Quantifiers based on ELM represent a family of methods based on structured output learning;
    these quantifiers rely on classifiers that have been optimized using a quantification-oriented loss
    measure. This implementation relies on
    `Joachims’ SVM perf <https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html>`_ structured output
    learning algorithm, which has to be installed and patched for the purpose (see this
    `script <https://github.com/HLT-ISTI/QuaPy/blob/master/prepare_svmperf.sh>`_).
    This function is a wrapper around CC(SVMperf(svmperf_base, loss, C))

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`; if set to None (default)
        this path will be obtained from qp.environ['SVMPERF_HOME']
    :param C: trade-off between training error and margin (default 0.01)
    :return: returns an instance of CC set to work with SVMperf (with loss and C set properly) as the
        underlying classifier
    """
    return newELM(svmperf_base, loss='q', C=C)


def newSVMKLD(svmperf_base=None, C=1):
    """
    SVM(KLD) is an Explicit Loss Minimization (ELM) quantifier set to optimize for the Kullback-Leibler Divergence
    as proposed by `Esuli et al. 2015 <https://dl.acm.org/doi/abs/10.1145/2700406>`_.
    Equivalent to:

    >>> CC(SVMperf(svmperf_base, loss='kld', C=C))

    Quantifiers based on ELM represent a family of methods based on structured output learning;
    these quantifiers rely on classifiers that have been optimized using a quantification-oriented loss
    measure. This implementation relies on
    `Joachims’ SVM perf <https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html>`_ structured output
    learning algorithm, which has to be installed and patched for the purpose (see this
    `script <https://github.com/HLT-ISTI/QuaPy/blob/master/prepare_svmperf.sh>`_).
    This function is a wrapper around CC(SVMperf(svmperf_base, loss, C))

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`; if set to None (default)
        this path will be obtained from qp.environ['SVMPERF_HOME']
    :param C: trade-off between training error and margin (default 0.01)
    :return: returns an instance of CC set to work with SVMperf (with loss and C set properly) as the
        underlying classifier
    """
    return newELM(svmperf_base, loss='kld', C=C)


def newSVMKLD(svmperf_base=None, C=1):
    """
    SVM(KLD) is an Explicit Loss Minimization (ELM) quantifier set to optimize for the Kullback-Leibler Divergence
    normalized via the logistic function, as proposed by
    `Esuli et al. 2015 <https://dl.acm.org/doi/abs/10.1145/2700406>`_.
    Equivalent to:

    >>> CC(SVMperf(svmperf_base, loss='nkld', C=C))

    Quantifiers based on ELM represent a family of methods based on structured output learning;
    these quantifiers rely on classifiers that have been optimized using a quantification-oriented loss
    measure. This implementation relies on
    `Joachims’ SVM perf <https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html>`_ structured output
    learning algorithm, which has to be installed and patched for the purpose (see this
    `script <https://github.com/HLT-ISTI/QuaPy/blob/master/prepare_svmperf.sh>`_).
    This function is a wrapper around CC(SVMperf(svmperf_base, loss, C))

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`; if set to None (default)
        this path will be obtained from qp.environ['SVMPERF_HOME']
    :param C: trade-off between training error and margin (default 0.01)
    :return: returns an instance of CC set to work with SVMperf (with loss and C set properly) as the
        underlying classifier
    """
    return newELM(svmperf_base, loss='nkld', C=C)


def newSVMAE(svmperf_base=None, C=1):
    """
    SVM(KLD) is an Explicit Loss Minimization (ELM) quantifier set to optimize for the Absolute Error as first used by
    `Moreo and Sebastiani, 2021 <https://arxiv.org/abs/2011.02552>`_.
    Equivalent to:

    >>> CC(SVMperf(svmperf_base, loss='mae', C=C))

    Quantifiers based on ELM represent a family of methods based on structured output learning;
    these quantifiers rely on classifiers that have been optimized using a quantification-oriented loss
    measure. This implementation relies on
    `Joachims’ SVM perf <https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html>`_ structured output
    learning algorithm, which has to be installed and patched for the purpose (see this
    `script <https://github.com/HLT-ISTI/QuaPy/blob/master/prepare_svmperf.sh>`_).
    This function is a wrapper around CC(SVMperf(svmperf_base, loss, C))

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`; if set to None (default)
        this path will be obtained from qp.environ['SVMPERF_HOME']
    :param C: trade-off between training error and margin (default 0.01)
    :return: returns an instance of CC set to work with SVMperf (with loss and C set properly) as the
        underlying classifier
    """
    return newELM(svmperf_base, loss='mae', C=C)


def newSVMRAE(svmperf_base=None, C=1):
    """
    SVM(KLD) is an Explicit Loss Minimization (ELM) quantifier set to optimize for the Relative Absolute Error as first
    used by `Moreo and Sebastiani, 2021 <https://arxiv.org/abs/2011.02552>`_.
    Equivalent to:

    >>> CC(SVMperf(svmperf_base, loss='mrae', C=C))

    Quantifiers based on ELM represent a family of methods based on structured output learning;
    these quantifiers rely on classifiers that have been optimized using a quantification-oriented loss
    measure. This implementation relies on
    `Joachims’ SVM perf <https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html>`_ structured output
    learning algorithm, which has to be installed and patched for the purpose (see this
    `script <https://github.com/HLT-ISTI/QuaPy/blob/master/prepare_svmperf.sh>`_).
    This function is a wrapper around CC(SVMperf(svmperf_base, loss, C))

    :param svmperf_base: path to the folder containing the binary files of `SVM perf`; if set to None (default)
        this path will be obtained from qp.environ['SVMPERF_HOME']
    :param C: trade-off between training error and margin (default 0.01)
    :return: returns an instance of CC set to work with SVMperf (with loss and C set properly) as the
        underlying classifier
    """
    return newELM(svmperf_base, loss='mrae', C=C)


class OneVsAllAggregative(OneVsAllGeneric, AggregativeQuantifier):
    """
    Allows any binary quantifier to perform quantification on single-label datasets.
    The method maintains one binary quantifier for each class, and then l1-normalizes the outputs so that the
    class prevelences sum up to 1.
    This variant was used, along with the :class:`EMQ` quantifier, in
    `Gao and Sebastiani, 2016 <https://link.springer.com/content/pdf/10.1007/s13278-016-0327-z.pdf>`_.

    :param binary_quantifier: a quantifier (binary) that will be employed to work on multiclass model in a
        one-vs-all manner
    :param n_jobs: number of parallel workers
    :param parallel_backend: the parallel backend for joblib (default "loky"); this is helpful for some quantifiers
        (e.g., ELM-based ones) that cannot be run with multiprocessing, since the temp dir they create during fit will
        is removed and no longer available at predict time.
    """

    def __init__(self, binary_quantifier, n_jobs=None, parallel_backend='multiprocessing'):
        assert isinstance(binary_quantifier, BaseQuantifier), \
            f'{self.binary_quantifier} does not seem to be a Quantifier'
        assert isinstance(binary_quantifier, AggregativeQuantifier), \
            f'{self.binary_quantifier} does not seem to be of type Aggregative'
        self.binary_quantifier = binary_quantifier
        self.n_jobs = qp._get_njobs(n_jobs)
        self.parallel_backend = parallel_backend

    def classify(self, X):
        """
        If the base quantifier is not probabilistic, returns a matrix of shape `(n,m,)` with `n` the number of
        instances and `m` the number of classes. The entry `(i,j)` is a binary value indicating whether instance
        `i `belongs to class `j`. The binary classifications are independent of each other, meaning that an instance
        can end up be attributed to 0, 1, or more classes.
        If the base quantifier is probabilistic, returns a matrix of shape `(n,m,2)` with `n` the number of instances
        and `m` the number of classes. The entry `(i,j,1)` (resp. `(i,j,0)`) is a value in [0,1] indicating the
        posterior probability that instance `i` belongs (resp. does not belong) to class `j`. The posterior
        probabilities are independent of each other, meaning that, in general, they do not sum up to one.

        :param X: array-like
        :return: `np.ndarray`
        """

        classif_predictions = self._parallel(self._delayed_binary_classification, X)
        if isinstance(self.binary_quantifier, AggregativeSoftQuantifier):
            return np.swapaxes(classif_predictions, 0, 1)
        else:
            return classif_predictions.T

    def aggregate(self, classif_predictions):
        prevalences = self._parallel(self._delayed_binary_aggregate, classif_predictions)
        return F.normalize_prevalence(prevalences)

    def _delayed_binary_classification(self, c, X):
        return self.dict_binary_quantifiers[c].classify(X)

    def _delayed_binary_aggregate(self, c, classif_predictions):
        # the estimation for the positive class prevalence
        return self.dict_binary_quantifiers[c].aggregate(classif_predictions[:, c])[1]


class AggregativeMedianEstimator(BinaryQuantifier):
    """
    This method is a meta-quantifier that returns, as the estimated class prevalence values, the median of the
    estimation returned by differently (hyper)parameterized base quantifiers.
    The median of unit-vectors is only guaranteed to be a unit-vector for n=2 dimensions,
    i.e., in cases of binary quantification.

    :param base_quantifier: the base, binary quantifier
    :param random_state: a seed to be set before fitting any base quantifier (default None)
    :param param_grid: the grid or parameters towards which the median will be computed
    :param n_jobs: number of parallel workers
    """

    def __init__(self, base_quantifier: AggregativeQuantifier, param_grid: dict, random_state=None, n_jobs=None):
        self.base_quantifier = base_quantifier
        self.param_grid = param_grid
        self.random_state = random_state
        self.n_jobs = qp._get_njobs(n_jobs)

    def get_params(self, deep=True):
        return self.base_quantifier.get_params(deep)

    def set_params(self, **params):
        self.base_quantifier.set_params(**params)

    def _delayed_fit(self, args):
        with qp.util.temp_seed(self.random_state):
            params, X, y = args
            model = deepcopy(self.base_quantifier)
            model.set_params(**params)
            model.fit(X, y)
            return model

    def _delayed_fit_classifier(self, args):
        with qp.util.temp_seed(self.random_state):
            cls_params, X, y = args
            model = deepcopy(self.base_quantifier)
            model.set_params(**cls_params)
            predictions, labels = model.classifier_fit_predict(X, y)
            return (model, predictions, labels)

    def _delayed_fit_aggregation(self, args):
        with qp.util.temp_seed(self.random_state):
            ((model, predictions, y), q_params) = args
            model = deepcopy(model)
            model.set_params(**q_params)
            model.aggregation_fit(predictions, y)
            return model

    def fit(self, X, y):
        import itertools

        self._check_binary(y, self.__class__.__name__)

        if isinstance(self.base_quantifier, AggregativeQuantifier):
            cls_configs, q_configs = qp.model_selection.group_params(self.param_grid)

            if len(cls_configs) > 1:
                models_preds = qp.util.parallel(
                    self._delayed_fit_classifier,
                    ((params, X, y) for params in cls_configs),
                    seed=qp.environ.get('_R_SEED', None),
                    n_jobs=self.n_jobs,
                    asarray=False,
                    backend='threading'
                )
            else:
                model = self.base_quantifier
                model.set_params(**cls_configs[0])
                predictions, labels = model.classifier_fit_predict(X, y)
                models_preds = [(model, predictions, labels)]

            self.models = qp.util.parallel(
                self._delayed_fit_aggregation,
                itertools.product(models_preds, q_configs),
                seed=qp.environ.get('_R_SEED', None),
                n_jobs=self.n_jobs,
                backend='threading'
            )
        else:
            configs = qp.model_selection.expand_grid(self.param_grid)
            self.models = qp.util.parallel(
                self._delayed_fit,
                ((params, X, y) for params in configs),
                seed=qp.environ.get('_R_SEED', None),
                n_jobs=self.n_jobs,
                backend='threading'
            )
        return self

    def _delayed_predict(self, args):
        model, instances = args
        return model.predict(instances)

    def predict(self, instances):
        prev_preds = qp.util.parallel(
            self._delayed_predict,
            ((model, instances) for model in self.models),
            seed=qp.environ.get('_R_SEED', None),
            n_jobs=self.n_jobs,
            backend='threading'
        )
        return np.median(prev_preds, axis=0)


# ---------------------------------------------------------------
# imports
# ---------------------------------------------------------------

from . import _threshold_optim

T50 = _threshold_optim.T50
MAX = _threshold_optim.MAX
X = _threshold_optim.X
MS = _threshold_optim.MS
MS2 = _threshold_optim.MS2

from . import _kdey

KDEyML = _kdey.KDEyML
KDEyHD = _kdey.KDEyHD
KDEyCS = _kdey.KDEyCS

# ---------------------------------------------------------------
# aliases
# ---------------------------------------------------------------

ClassifyAndCount = CC
AdjustedClassifyAndCount = ACC
ProbabilisticClassifyAndCount = PCC
ProbabilisticAdjustedClassifyAndCount = PACC
ExpectationMaximizationQuantifier = EMQ
SLD = EMQ
DistributionMatchingY = DMy
HellingerDistanceY = HDy
MedianSweep = MS
MedianSweep2 = MS2
