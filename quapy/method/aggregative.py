from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Literal, Union
import numpy as np
from abstention.calibration import NoBiasVectorScaling, TempScaling, VectorScaling
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

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
    """

    val_split_ = None

    @property
    def val_split(self):
        return self.val_split_

    @val_split.setter
    def val_split(self, val_split):
        if isinstance(val_split, LabelledCollection):
            print('warning: setting val_split with a LabelledCollection will be inefficient in'
                  'model selection. Rather pass the LabelledCollection at fit time')
        self.val_split_ = val_split

    def _check_init_parameters(self):
        """
        Implements any check to be performed in the parameters of the init method before undertaking
        the training of the quantifier. This is made as to allow for a quick execution stop when the
        parameters are not valid.

        :return: Nothing. May raise an exception.
        """
        pass

    def _check_non_empty_classes(self, data: LabelledCollection):
        """
        Asserts all classes have positive instances.

        :param data: LabelledCollection
        :return: Nothing. May raise an exception.
        """
        sample_prevs = data.prevalence()
        empty_classes = np.argwhere(sample_prevs==0).flatten()
        if len(empty_classes)>0:
            empty_class_names = data.classes_[empty_classes]
            raise ValueError(f'classes {empty_class_names} have no training examples')

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split=None):
        """
        Trains the aggregative quantifier. This comes down to training a classifier and an aggregation function.

        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
        :param fit_classifier: whether to train the learner (default is True). Set to False if the
            learner has been trained outside the quantifier.
        :param val_split: specifies the data used for generating classifier predictions. This specification
            can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
            be extracted from the training set; or as an integer (default 5), indicating that the predictions
            are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
            for `k`); or as a collection defining the specific set of data to use for validation.
            Alternatively, this set can be specified at fit time by indicating the exact set of data
            on which the predictions are to be generated.
        :return: self
        """
        self._check_init_parameters()
        classif_predictions = self.classifier_fit_predict(data, fit_classifier, predict_on=val_split)
        self.aggregation_fit(classif_predictions, data)
        return self

    def classifier_fit_predict(self, data: LabelledCollection, fit_classifier=True, predict_on=None):
        """
        Trains the classifier if requested (`fit_classifier=True`) and generate the necessary predictions to
        train the aggregation function.

        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
        :param fit_classifier: whether to train the learner (default is True). Set to False if the
            learner has been trained outside the quantifier.
        :param predict_on: specifies the set on which predictions need to be issued. This parameter can
            be specified as None (default) to indicate no prediction is needed; a float in (0, 1) to
            indicate the proportion of instances to be used for predictions (the remainder is used for
            training); an integer >1 to indicate that the predictions must be generated via k-fold
            cross-validation, using this integer as k; or the data sample itself on which to generate
            the predictions.
        """
        assert isinstance(fit_classifier, bool), 'unexpected type for "fit_classifier", must be boolean'

        self._check_classifier(adapt_if_necessary=(self._classifier_method() == 'predict_proba'))

        if fit_classifier:
            self._check_non_empty_classes(data)

        if predict_on is None:
            if not fit_classifier:
                predict_on = data
                if isinstance(self.val_split, LabelledCollection) and self.val_split!=predict_on:
                    raise ValueError(f'{fit_classifier=} but a LabelledCollection was provided as val_split '
                                     f'in __init__ that is not the same as the LabelledCollection provided in fit.')
        if predict_on is None:
            predict_on = self.val_split

        if predict_on is None:
            if fit_classifier:
                self.classifier.fit(*data.Xy)
            predictions = None
        elif isinstance(predict_on, float):
            if fit_classifier:
                if not (0. < predict_on < 1.):
                    raise ValueError(f'proportion {predict_on=} out of range, must be in (0,1)')
                train, val = data.split_stratified(train_prop=(1 - predict_on))
                self.classifier.fit(*train.Xy)
                predictions = LabelledCollection(self.classify(val.X), val.y, classes=data.classes_)
            else:
                raise ValueError(f'wrong type for predict_on: since fit_classifier=False, '
                                 f'the set on which predictions have to be issued must be '
                                 f'explicitly indicated')

        elif isinstance(predict_on, LabelledCollection):
            if fit_classifier:
                self.classifier.fit(*data.Xy)
            predictions = LabelledCollection(self.classify(predict_on.X), predict_on.y, classes=predict_on.classes_)

        elif isinstance(predict_on, int):
            if fit_classifier:
                if predict_on <= 1:
                    raise ValueError(f'invalid value {predict_on} in fit. '
                                     f'Specify a integer >1 for kFCV estimation.')
                else:
                    n_jobs = self.n_jobs if hasattr(self, 'n_jobs') else qp._get_njobs(None)
                    predictions = cross_val_predict(
                        self.classifier, *data.Xy, cv=predict_on, n_jobs=n_jobs, method=self._classifier_method())
                    predictions = LabelledCollection(predictions, data.y, classes=data.classes_)
                    self.classifier.fit(*data.Xy)
            else:
                raise ValueError(f'wrong type for predict_on: since fit_classifier=False, '
                                 f'the set on which predictions have to be issued must be '
                                 f'explicitly indicated')

        else:
            raise ValueError(
                f'error: param "predict_on" ({type(predict_on)}) not understood; '
                f'use either a float indicating the split proportion, or a '
                f'tuple (X,y) indicating the validation partition')

        return predictions

    @abstractmethod
    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        """
        Trains the aggregation function.

        :param classif_predictions: a :class:`quapy.data.base.LabelledCollection` containing,
            as instances, the predictions issued by the classifier and, as labels, the true labels
        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
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

    def classify(self, instances):
        """
        Provides the label predictions for the given instances. The predictions should respect the format expected by
        :meth:`aggregate`, e.g., posterior probabilities for probabilistic quantifiers, or crisp predictions for
        non-probabilistic quantifiers. The default one is "decision_function".

        :param instances: array-like of shape `(n_instances, n_features,)`
        :return: np.ndarray of shape `(n_instances,)` with label predictions
        """
        return getattr(self.classifier, self._classifier_method())(instances)

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

        :param adapt_if_necessary: if True, the method will try to comply with the required specifications
        """
        assert hasattr(self.classifier, self._classifier_method()), \
            f"the method does not implement the required {self._classifier_method()} method"

    def quantify(self, instances):
        """
        Generate class prevalence estimates for the sample's instances by aggregating the label predictions generated
        by the classifier.

        :param instances: array-like
        :return: `np.ndarray` of shape `(n_classes)` with class prevalence estimates.
        """
        classif_predictions = self.classify(instances)
        return self.aggregate(classif_predictions)

    @abstractmethod
    def aggregate(self, classif_predictions: np.ndarray):
        """
        Implements the aggregation of label predictions.

        :param classif_predictions: `np.ndarray` of label predictions
        :return: `np.ndarray` of shape `(n_classes,)` with class prevalence estimates.
        """
        ...

    @property
    def classes_(self):
        """
        Class labels, in the same order in which class prevalence values are to be computed.
        This default implementation actually returns the class labels of the learner.

        :return: array-like
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

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split=None):
        self._check_binary(data, self.__class__.__name__)
        return super().fit(data, fit_classifier, val_split)


# Methods
# ------------------------------------
class CC(AggregativeCrispQuantifier):
    """
    The most basic Quantification method. One that simply classifies all instances and counts how many have been
    attributed to each of the classes in order to compute class prevalence estimates.

    :param classifier: a sklearn's Estimator that generates a classifier
    """

    def __init__(self, classifier: BaseEstimator=None):
        self.classifier = qp._get_classifier(classifier)

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        """
        Nothing to do here!

        :param classif_predictions: not used
        :param data: not used
        """
        pass

    def aggregate(self, classif_predictions: np.ndarray):
        """
        Computes class prevalence estimates by counting the prevalence of each of the predicted labels.

        :param classif_predictions: array-like with label predictions
        :return: `np.ndarray` of shape `(n_classes,)` with class prevalence estimates.
        """
        return F.prevalence_from_labels(classif_predictions, self.classes_)


class PCC(AggregativeSoftQuantifier):
    """
    `Probabilistic Classify & Count <https://ieeexplore.ieee.org/abstract/document/5694031>`_,
    the probabilistic variant of CC that relies on the posterior probabilities returned by a probabilistic classifier.

    :param classifier: a sklearn's Estimator that generates a classifier
    """

    def __init__(self, classifier: BaseEstimator=None):
        self.classifier = qp._get_classifier(classifier)

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        """
        Nothing to do here!

        :param classif_predictions: not used
        :param data: not used
        """
        pass

    def aggregate(self, classif_posteriors):
        return F.prevalence_from_probabilities(classif_posteriors, binarize=False)


class ACC(AggregativeCrispQuantifier):
    """
    `Adjusted Classify & Count <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_,
    the "adjusted" variant of :class:`CC`, that corrects the predictions of CC
    according to the `misclassification rates`.

    :param classifier: a sklearn's Estimator that generates a classifier

    :param val_split: specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer (default 5), indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`); or as a collection defining the specific set of data to use for validation.
        Alternatively, this set can be specified at fit time by indicating the exact set of data
        on which the predictions are to be generated.

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
            classifier: BaseEstimator=None,
            val_split=5,
            solver: Literal['minimize', 'exact', 'exact-raise', 'exact-cc'] = 'minimize',
            method: Literal['inversion', 'invariant-ratio'] = 'inversion',
            norm: Literal['clip', 'mapsimplex', 'condsoftmax'] = 'clip',
            n_jobs=None,
    ):
        self.classifier = qp._get_classifier(classifier)
        self.val_split = val_split
        self.n_jobs = qp._get_njobs(n_jobs)
        self.solver = solver
        self.method = method
        self.norm = norm

    SOLVERS = ['exact', 'minimize', 'exact-raise', 'exact-cc']
    METHODS = ['inversion', 'invariant-ratio']
    NORMALIZATIONS = ['clip', 'mapsimplex', 'condsoftmax', None]

    @classmethod
    def newInvariantRatioEstimation(cls, classifier: BaseEstimator, val_split=5, n_jobs=None):
        """
        Constructs a quantifier that implements the Invariant Ratio Estimator of
        `Vaz et al. 2018 <https://jmlr.org/papers/v20/18-456.html>`_. This amounts
        to setting method to 'invariant-ratio' and clipping to 'project'.

        :param classifier: a sklearn's Estimator that generates a classifier
        :param val_split: specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer (default 5), indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`); or as a collection defining the specific set of data to use for validation.
        Alternatively, this set can be specified at fit time by indicating the exact set of data
        on which the predictions are to be generated.
        :param n_jobs: number of parallel workers
        :return: an instance of ACC configured so that it implements the Invariant Ratio Estimator
        """
        return ACC(classifier, val_split=val_split, method='invariant-ratio', norm='mapsimplex', n_jobs=n_jobs)

    def _check_init_parameters(self):
        if self.solver not in ACC.SOLVERS:
            raise ValueError(f"unknown solver; valid ones are {ACC.SOLVERS}")
        if self.method not in ACC.METHODS:
            raise ValueError(f"unknown method; valid ones are {ACC.METHODS}")
        if self.norm not in ACC.NORMALIZATIONS:
            raise ValueError(f"unknown normalization; valid ones are {ACC.NORMALIZATIONS}")

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        """
        Estimates the misclassification rates.

        :param classif_predictions: a :class:`quapy.data.base.LabelledCollection` containing,
            as instances, the label predictions issued by the classifier and, as labels, the true labels
        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
        """
        pred_labels, true_labels = classif_predictions.Xy
        self.cc = CC(self.classifier)
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

    :param classifier: a sklearn's Estimator that generates a classifier

    :param val_split: specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer (default 5), indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`). Alternatively, this set can be specified at fit time by indicating the exact set of data
        on which the predictions are to be generated.

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
            classifier: BaseEstimator=None,
            val_split=5,
            solver: Literal['minimize', 'exact', 'exact-raise', 'exact-cc'] = 'minimize',
            method: Literal['inversion', 'invariant-ratio'] = 'inversion',
            norm: Literal['clip', 'mapsimplex', 'condsoftmax'] = 'clip',
            n_jobs=None
    ):
        self.classifier = qp._get_classifier(classifier)
        self.val_split = val_split
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


    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        """
        Estimates the misclassification rates

        :param classif_predictions: a :class:`quapy.data.base.LabelledCollection` containing,
            as instances, the posterior probabilities issued by the classifier and, as labels, the true labels
        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
        """
        posteriors, true_labels = classif_predictions.Xy
        self.pcc = PCC(self.classifier)
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

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: specifies the data used for generating classifier predictions. This specification
        can be made as float in (0, 1) indicating the proportion of stratified held-out validation set to
        be extracted from the training set; or as an integer, indicating that the predictions
        are to be generated in a `k`-fold cross-validation manner (with this integer indicating the value
        for `k`, default 5); or as a collection defining the specific set of data to use for validation.
        Alternatively, this set can be specified at fit time by indicating the exact set of data
        on which the predictions are to be generated. This hyperparameter is only meant to be used when the
        heuristics are to be applied, i.e., if a recalibration is required. The default value is None (meaning
        the recalibration is not required). In case this hyperparameter is set to a value other than None, but
        the recalibration is not required (recalib=None), a warning message will be raised.
    :param exact_train_prev: set to True (default) for using the true training prevalence as the initial observation;
        set to False for computing the training prevalence as an estimate of it, i.e., as the expected
        value of the posterior probabilities of the training instances.
    :param recalib: a string indicating the method of recalibration.
        Available choices include "nbvs" (No-Bias Vector Scaling), "bcts" (Bias-Corrected Temperature Scaling,
        default), "ts" (Temperature Scaling), and "vs" (Vector Scaling). Default is None (no recalibration).
    :param n_jobs: number of parallel workers. Only used for recalibrating the classifier if `val_split` is set to
        an integer `k` --the number of folds.
    """

    MAX_ITER = 1000
    EPSILON = 1e-4

    def __init__(self, classifier: BaseEstimator=None, val_split=None, exact_train_prev=True, recalib=None, n_jobs=None):
        self.classifier = qp._get_classifier(classifier)
        self.val_split = val_split
        self.exact_train_prev = exact_train_prev
        self.recalib = recalib
        self.n_jobs = n_jobs

    @classmethod
    def EMQ_BCTS(cls, classifier: BaseEstimator, n_jobs=None):
        """
        Constructs an instance of EMQ using the best configuration found in the `Alexandari et al. paper
        <http://proceedings.mlr.press/v119/alexandari20a.html>`_, i.e., one that relies on Bias-Corrected Temperature
        Scaling (BCTS) as a recalibration function, and that uses an estimate of the training prevalence instead of
        the true training prevalence.

        :param classifier: a sklearn's Estimator that generates a classifier
        :param n_jobs: number of parallel workers.
        :return: An instance of EMQ with BCTS
        """
        return EMQ(classifier, val_split=5, exact_train_prev=False, recalib='bcts', n_jobs=n_jobs)

    def _check_init_parameters(self):
        if self.val_split is not None:
            if self.exact_train_prev and self.recalib is None:
                raise RuntimeWarning(f'The parameter {self.val_split=} was specified for EMQ, while the parameters '
                      f'{self.exact_train_prev=} and {self.recalib=}. This has no effect and causes an unnecessary '
                      f'overload.')
        else:
            if self.recalib is not None:
                print(f'[warning] The parameter {self.recalib=} requires the val_split be different from None. '
                      f'This parameter will be set to 5. To avoid this warning, set this value to a float value '
                      f'indicating the proportion of training data to be used as validation, or to an integer '
                      f'indicating the number of folds for kFCV.')
                self.val_split=5

    def classify(self, instances):
        """
        Provides the posterior probabilities for the given instances. If the classifier was required
        to be recalibrated, then these posteriors are recalibrated accordingly.

        :param instances: array-like of shape `(n_instances, n_dimensions,)`
        :return: np.ndarray of shape `(n_instances, n_classes,)` with posterior probabilities
        """
        posteriors = self.classifier.predict_proba(instances)
        if hasattr(self, 'calibration_function') and self.calibration_function is not None:
            posteriors = self.calibration_function(posteriors)
        return posteriors

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        """
        Trains the aggregation function of EMQ. This comes down to recalibrating the posterior probabilities
        ir requested.

        :param classif_predictions: a :class:`quapy.data.base.LabelledCollection` containing,
            as instances, the posterior probabilities issued by the classifier and, as labels, the true labels
        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
        """
        if self.recalib is not None:
            P, y = classif_predictions.Xy
            if self.recalib == 'nbvs':
                calibrator = NoBiasVectorScaling()
            elif self.recalib == 'bcts':
                calibrator = TempScaling(bias_positions='all')
            elif self.recalib == 'ts':
                calibrator = TempScaling()
            elif self.recalib == 'vs':
                calibrator = VectorScaling()
            else:
                raise ValueError('invalid param argument for recalibration method; available ones are '
                                 '"nbvs", "bcts", "ts", and "vs".')

            if not np.issubdtype(y.dtype, np.number):
                y = np.searchsorted(data.classes_, y)
            self.calibration_function = calibrator(P, np.eye(data.n_classes)[y], posterior_supplied=True)

        if self.exact_train_prev:
            self.train_prevalence = data.prevalence()
        else:
            train_posteriors = classif_predictions.X
            if self.recalib is not None:
                train_posteriors = self.calibration_function(train_posteriors)
            self.train_prevalence = F.prevalence_from_probabilities(train_posteriors)

    def aggregate(self, classif_posteriors, epsilon=EPSILON):
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


class BayesianCC(AggregativeCrispQuantifier):
    """
    `Bayesian quantification <https://arxiv.org/abs/2302.09159>`_ method,
    which is a variant of :class:`ACC` that calculates the posterior probability distribution
    over the prevalence vectors, rather than providing a point estimate obtained
    by matrix inversion.

    Can be used to diagnose degeneracy in the predictions visible when the confusion
    matrix has high condition number or to quantify uncertainty around the point estimate.

    This method relies on extra dependencies, which have to be installed via:
    `$ pip install quapy[bayes]`

    :param classifier: a sklearn's Estimator that generates a classifier
    :param val_split: a float in (0, 1) indicating the proportion of the training data to be used,
        as a stratified held-out validation set, for generating classifier predictions.
    :param num_warmup: number of warmup iterations for the MCMC sampler (default 500)
    :param num_samples: number of samples to draw from the posterior (default 1000)
    :param mcmc_seed: random seed for the MCMC sampler (default 0)
    """
    def __init__(self,
                 classifier: BaseEstimator=None,
                 val_split: float = 0.75,
                 num_warmup: int = 500,
                 num_samples: int = 1_000,
                 mcmc_seed: int = 0):

        if num_warmup <= 0:
            raise ValueError(f'parameter {num_warmup=} must be a positive integer')
        if num_samples <= 0:
            raise ValueError(f'parameter {num_samples=} must be a positive integer')

        if (not isinstance(val_split, float)) or val_split <= 0 or val_split >= 1:
            raise ValueError(f'val_split must be a float in (0, 1), got {val_split}')

        if _bayesian.DEPENDENCIES_INSTALLED is False:
            raise ImportError("Auxiliary dependencies are required. Run `$ pip install quapy[bayes]` to install them.")

        self.classifier = qp._get_classifier(classifier)
        self.val_split = val_split
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.mcmc_seed = mcmc_seed

        # Array of shape (n_classes, n_predicted_classes,) where entry (y, c) is the number of instances
        # labeled as class y and predicted as class c.
        # By default, this array is set to None and later defined as part of the `aggregation_fit` phase
        self._n_and_c_labeled = None

        # Dictionary with posterior samples, set when `aggregate` is provided.
        self._samples = None

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        """
        Estimates the misclassification rates.

        :param classif_predictions: a :class:`quapy.data.base.LabelledCollection` containing,
            as instances, the label predictions issued by the classifier and, as labels, the true labels
        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
        """
        pred_labels, true_labels = classif_predictions.Xy
        self._n_and_c_labeled = confusion_matrix(y_true=true_labels, y_pred=pred_labels, labels=self.classifier.classes_)

    def sample_from_posterior(self, classif_predictions):
        if self._n_and_c_labeled is None:
            raise ValueError("aggregation_fit must be called before sample_from_posterior")

        n_c_unlabeled = F.counts_from_labels(classif_predictions, self.classifier.classes_)

        self._samples = _bayesian.sample_posterior(
            n_c_unlabeled=n_c_unlabeled,
            n_y_and_c_labeled=self._n_and_c_labeled,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            seed=self.mcmc_seed,
        )
        return self._samples

    def get_prevalence_samples(self):
        if self._samples is None:
            raise ValueError("sample_from_posterior must be called before get_prevalence_samples")
        return self._samples[_bayesian.P_TEST_Y]

    def get_conditional_probability_samples(self):
        if self._samples is None:
            raise ValueError("sample_from_posterior must be called before get_conditional_probability_samples")
        return self._samples[_bayesian.P_C_COND_Y]

    def aggregate(self, classif_predictions):
        samples = self.sample_from_posterior(classif_predictions)[_bayesian.P_TEST_Y]
        return np.asarray(samples.mean(axis=0), dtype=float)


class HDy(AggregativeSoftQuantifier, BinaryAggregativeQuantifier):
    """
    `Hellinger Distance y <https://www.sciencedirect.com/science/article/pii/S0020025512004069>`_ (HDy).
    HDy is a probabilistic method for training binary quantifiers, that models quantification as the problem of
    minimizing the divergence (in terms of the Hellinger Distance) between two distributions of posterior
    probabilities returned by the classifier. One of the distributions is generated from the unlabelled examples and
    the other is generated from a validation set. This latter distribution is defined as a mixture of the
    class-conditional distributions of the posterior probabilities returned for the positive and negative validation
    examples, respectively. The parameters of the mixture thus represent the estimates of the class prevalence values.

    :param classifier: a sklearn's Estimator that generates a binary classifier
    :param val_split: a float in range (0,1) indicating the proportion of data to be used as a stratified held-out
        validation distribution, or a :class:`quapy.data.base.LabelledCollection` (the split itself), or an integer indicating the number of folds (default 5)..
    """

    def __init__(self, classifier: BaseEstimator=None, val_split=5):
        self.classifier = qp._get_classifier(classifier)
        self.val_split = val_split

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        """
        Trains the aggregation function of HDy.

        :param classif_predictions: a :class:`quapy.data.base.LabelledCollection` containing,
            as instances, the posterior probabilities issued by the classifier and, as labels, the true labels
        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
        """
        P, y = classif_predictions.Xy
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

    :param classifier: a sklearn's Estimator that generates a binary classifier
    :param val_split: a float in range (0,1) indicating the proportion of data to be used as a stratified held-out
        validation distribution, or a :class:`quapy.data.base.LabelledCollection` (the split itself), or an integer indicating the number of folds (default 5)..
    :param n_bins: an int with the number of bins to use to compute the histograms.
    :param divergence: a str indicating the name of divergence (currently supported ones are "HD" or "topsoe"), or a
        callable function computes the divergence between two distributions (two equally sized arrays).
    :param tol: a float with the tolerance for the ternary search algorithm.
    :param n_jobs: number of parallel workers.
    """

    def __init__(self, classifier: BaseEstimator=None, val_split=5, n_bins=8, divergence: Union[str, Callable]= 'HD', tol=1e-05, n_jobs=None):
        self.classifier = qp._get_classifier(classifier)
        self.val_split = val_split
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

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        """
        Trains the aggregation function of DyS.

        :param classif_predictions: a :class:`quapy.data.base.LabelledCollection` containing,
            as instances, the posterior probabilities issued by the classifier and, as labels, the true labels
        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
        """
        Px, y = classif_predictions.Xy
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

    :param classifier: a sklearn's Estimator that generates a binary classifier.
    :param val_split: a float in range (0,1) indicating the proportion of data to be used as a stratified held-out
        validation distribution, or a :class:`quapy.data.base.LabelledCollection` (the split itself), or an integer indicating the number of folds (default 5)..
    """

    def __init__(self, classifier: BaseEstimator=None, val_split=5):
        self.classifier = qp._get_classifier(classifier)
        self.val_split = val_split
      
    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        """
        Trains the aggregation function of SMM.

        :param classif_predictions: a :class:`quapy.data.base.LabelledCollection` containing,
            as instances, the posterior probabilities issued by the classifier and, as labels, the true labels
        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
        """
        Px, y = classif_predictions.Xy
        Px = Px[:, self.pos_label]  # takes only the P(y=+1|x)
        self.Pxy1 = Px[y == self.pos_label]
        self.Pxy0 = Px[y == self.neg_label]
        self.Pxy1_mean = np.mean(self.Pxy1)  # equiv. TPR 
        self.Pxy0_mean = np.mean(self.Pxy0)  # equiv. FPR
        return self

    def aggregate(self, classif_posteriors):
        Px = classif_posteriors[:, self.pos_label]  # takes only the P(y=+1|x)
        Px_mean = np.mean(Px)
     
        class1_prev = (Px_mean - self.Pxy0_mean)/(self.Pxy1_mean - self.Pxy0_mean)
        return F.as_binary_prevalence(class1_prev, clip_if_necessary=True)


class DMy(AggregativeSoftQuantifier):
    """
    Generic Distribution Matching quantifier for binary or multiclass quantification based on the space of posterior
    probabilities. This implementation takes the number of bins, the divergence, and the possibility to work on CDF
    as hyperparameters.

    :param classifier: a `sklearn`'s Estimator that generates a probabilistic classifier
    :param val_split: indicates the proportion of data to be used as a stratified held-out validation set to model the
        validation distribution.
        This parameter can be indicated as a real value (between 0 and 1), representing a proportion of
        validation data, or as an integer, indicating that the validation distribution should be estimated via
        `k`-fold cross validation (this integer stands for the number of folds `k`, defaults 5), or as a
        :class:`quapy.data.base.LabelledCollection` (the split itself).
    :param nbins: number of bins used to discretize the distributions (default 8)
    :param divergence: a string representing a divergence measure (currently, "HD" and "topsoe" are implemented)
        or a callable function taking two ndarrays of the same dimension as input (default "HD", meaning Hellinger
        Distance)
    :param cdf: whether to use CDF instead of PDF (default False)
    :param n_jobs: number of parallel workers (default None)
    """

    def __init__(self, classifier: BaseEstimator=None, val_split=5, nbins=8, divergence: Union[str, Callable]='HD',
                 cdf=False, search='optim_minimize', n_jobs=None):
        self.classifier = qp._get_classifier(classifier)
        self.val_split = val_split
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
        distributions = counts/counts.sum(axis=1)[:,np.newaxis]
        if self.cdf:
            distributions = np.cumsum(distributions, axis=1)
        return distributions

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        """
        Trains the aggregation function of a distribution matching method. This comes down to generating the
        validation distributions out of the training data.
        The validation distributions have shape `(n, ch, nbins)`, with `n` the number of classes, `ch` the number of
        channels, and `nbins` the number of bins. In particular, let `V` be the validation distributions; then `di=V[i]`
        are the distributions obtained from training data labelled with class `i`; while `dij = di[j]` is the discrete
        distribution of posterior probabilities `P(Y=j|X=x)` for training data labelled with class `i`, and `dij[k]`
        is the fraction of instances with a value in the `k`-th bin.

        :param classif_predictions: a :class:`quapy.data.base.LabelledCollection` containing,
            as instances, the posterior probabilities issued by the classifier and, as labels, the true labels
        :param data: a :class:`quapy.data.base.LabelledCollection` consisting of the training data
        """
        posteriors, true_labels = classif_predictions.Xy
        n_classes = len(self.classifier.classes_)

        self.validation_distribution = qp.util.parallel(
            func=self._get_distributions,
            args=[posteriors[true_labels==cat] for cat in range(n_classes)],
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
            mixture_distribution = (prev @ self.validation_distribution.reshape(n_classes,-1)).reshape(n_channels, -1)
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

    def classify(self, instances):
        """
        If the base quantifier is not probabilistic, returns a matrix of shape `(n,m,)` with `n` the number of
        instances and `m` the number of classes. The entry `(i,j)` is a binary value indicating whether instance
        `i `belongs to class `j`. The binary classifications are independent of each other, meaning that an instance
        can end up be attributed to 0, 1, or more classes.
        If the base quantifier is probabilistic, returns a matrix of shape `(n,m,2)` with `n` the number of instances
        and `m` the number of classes. The entry `(i,j,1)` (resp. `(i,j,0)`) is a value in [0,1] indicating the
        posterior probability that instance `i` belongs (resp. does not belong) to class `j`. The posterior
        probabilities are independent of each other, meaning that, in general, they do not sum up to one.

        :param instances: array-like
        :return: `np.ndarray`
        """

        classif_predictions = self._parallel(self._delayed_binary_classification, instances)
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
    :param n_jobs: number of parllel workes
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
            params, training = args
            model = deepcopy(self.base_quantifier)
            model.set_params(**params)
            model.fit(training)
            return model

    def _delayed_fit_classifier(self, args):
        with qp.util.temp_seed(self.random_state):
            cls_params, training, kwargs = args
            model = deepcopy(self.base_quantifier)
            model.set_params(**cls_params)
            predictions = model.classifier_fit_predict(training, **kwargs)
            return (model, predictions)

    def _delayed_fit_aggregation(self, args):
        with qp.util.temp_seed(self.random_state):
            ((model, predictions), q_params), training = args
            model = deepcopy(model)
            model.set_params(**q_params)
            model.aggregation_fit(predictions, training)
            return model


    def fit(self, training: LabelledCollection, **kwargs):
        import itertools

        self._check_binary(training, self.__class__.__name__)

        if isinstance(self.base_quantifier, AggregativeQuantifier):
            cls_configs, q_configs = qp.model_selection.group_params(self.param_grid)

            if len(cls_configs) > 1:
                models_preds = qp.util.parallel(
                    self._delayed_fit_classifier,
                    ((params, training, kwargs) for params in cls_configs),
                    seed=qp.environ.get('_R_SEED', None),
                    n_jobs=self.n_jobs,
                    asarray=False,
                    backend='threading'
                )
            else:
                model = self.base_quantifier
                model.set_params(**cls_configs[0])
                predictions = model.classifier_fit_predict(training, **kwargs)
                models_preds = [(model, predictions)]

            self.models = qp.util.parallel(
                self._delayed_fit_aggregation,
                ((setup, training) for setup in itertools.product(models_preds, q_configs)),
                seed=qp.environ.get('_R_SEED', None),
                n_jobs=self.n_jobs,
                backend='threading'
            )
        else:
            configs = qp.model_selection.expand_grid(self.param_grid)
            self.models = qp.util.parallel(
                self._delayed_fit,
                ((params, training) for params in configs),
                seed=qp.environ.get('_R_SEED', None),
                n_jobs=self.n_jobs,
                backend='threading'
            )
        return self

    def _delayed_predict(self, args):
        model, instances = args
        return model.quantify(instances)

    def quantify(self, instances):
        prev_preds = qp.util.parallel(
            self._delayed_predict,
            ((model, instances) for model in self.models),
            seed=qp.environ.get('_R_SEED', None),
            n_jobs=self.n_jobs,
            backend='threading'
        )
        return np.median(prev_preds, axis=0)


#---------------------------------------------------------------
# imports
#---------------------------------------------------------------

from . import _threshold_optim

T50 = _threshold_optim.T50
MAX = _threshold_optim.MAX
X   = _threshold_optim.X
MS  = _threshold_optim.MS
MS2 = _threshold_optim.MS2


from . import _kdey

KDEyML = _kdey.KDEyML
KDEyHD = _kdey.KDEyHD
KDEyCS = _kdey.KDEyCS

#---------------------------------------------------------------
# aliases
#---------------------------------------------------------------

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
