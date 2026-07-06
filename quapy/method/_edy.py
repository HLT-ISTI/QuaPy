from typing import Callable, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

import quapy as qp
import quapy.functional as F
from quapy.method._helper import _get_quadprog
from quapy.method.aggregative import AggregativeSoftQuantifier


class EDy(AggregativeSoftQuantifier):
    """
    Energy Distance y (EDy), a posterior-space distribution-matching quantifier
    based on energy distance.

    The method represents each class by the posterior-probability vectors
    produced by a probabilistic classifier on validation data, and estimates the
    test prevalence vector by matching the test posterior distribution against
    the class-conditional validation distributions through an energy-distance
    objective solved as a quadratic program. The method is therefore another
    instance of the general mixture-matching view of quantification, but it
    operates directly on posterior vectors rather than on histogram summaries.

    This implementation works for binary and multiclass single-label
    quantification and relies on the optional ``quadprog`` dependency. It was
    adapted to QuaPy's current aggregative API from the original implementation
    available in `quantificationlib <https://github.com/AICGijon/quantificationlib>`_.

    The current implementation follows the energy-distance formulation discussed
    in:

    * Alberto Castaño, Laura Morán-Fernández, Jaime Alonso,
      Verónica Bolón-Canedo, Amparo Alonso-Betanzos, and Juan José del Coz.
      *An analysis of quantification methods based on matching distributions*.
    * Hideko Kawakubo, Marthinus Christoffel du Plessis, and Masashi Sugiyama
      (2016). *Computationally efficient class-prior estimation under class
      balance change using energy distance*. IEICE Transactions on Information
      and Systems, 99(1):176-186.

    :param classifier: a scikit-learn ``BaseEstimator``, or ``None`` to use
        ``qp.environ['DEFAULT_CLS']``
    :param fit_classifier: whether to train the learner (default ``True``).
        Set to ``False`` if the learner has already been trained outside the
        quantifier
    :param val_split: specification of the data used for generating validation
        posterior probabilities. This can be an integer (default ``5``) for
        k-fold cross-validation, a float in ``(0, 1)`` for a held-out split,
        or a tuple ``(X, y)`` with explicit validation data
    :param distance: distance used to compare posterior vectors. Valid string
        aliases are ``'manhattan'`` (default) and ``'euclidean'``; a custom
        callable compatible with pairwise-distance signatures can also be used
    :param n_jobs: number of parallel workers (default ``None``, meaning the
        value is taken from the environment)
    """

    def __init__(
        self,
        classifier: BaseEstimator = None,
        fit_classifier: bool = True,
        val_split=5,
        distance: Union[str, Callable] = 'manhattan',
        n_jobs=None,
    ):
        super().__init__(classifier, fit_classifier, val_split)
        self.distance = distance
        self.n_jobs = qp._get_njobs(n_jobs)
        self.train_n_cls_i_ = None
        self.train_distrib_ = None
        self.K_ = None
        self.G_ = None
        self.C_ = None
        self.b_ = None
        self.a_ = None

    def _check_init_parameters(self):
        self.distance = self._resolve_distance_function(self.distance)

    @staticmethod
    def _resolve_distance_function(distance):
        if isinstance(distance, str):
            if distance == 'manhattan':
                return manhattan_distances
            if distance == 'euclidean':
                return euclidean_distances
            raise ValueError(
                f"unknown distance {distance!r}; valid aliases are 'manhattan' and 'euclidean'"
            )
        if not hasattr(distance, '__call__'):
            raise ValueError('distance must be a valid string alias or a callable function')
        return distance

    def _is_pd(self, m):
        """Check whether a symmetric matrix is positive definite.

        This helper is used before invoking ``quadprog`` because the quadratic
        term of the optimization problem must be positive definite.
        """
        return self._dpofa(m)[0] == 0

    def _dpofa(self, m):
        """Factor a symmetric positive definite matrix.

        This is a lightweight Python adaptation of the ``dpofa`` routine used by
        ``quadprog``. Here it is mainly employed as a numerical check while
        preparing the quadratic-program matrix.
        """
        r = np.array(m, copy=True)
        n = len(r)
        for k in range(n):
            s = 0.0
            if k >= 1:
                for i in range(k):
                    t = r[i, k]
                    if i > 0:
                        t = t - np.sum(r[0:i, i] * r[0:i, k])
                    t = t / r[i, i]
                    r[i, k] = t
                    s = s + t * t
            s = r[k, k] - s
            if s <= 0.0:
                return k + 1, r
            r[k, k] = np.sqrt(s)
        return 0, r

    def _nearest_pd(self, A):
        """Project a matrix onto the cone of positive-definite matrices.

        In some cases the matrix induced by the energy-distance objective is not
        numerically positive definite, even though the underlying optimization
        problem is well posed. In those cases we replace it with the nearest
        positive-definite approximation before calling ``quadprog``.
        """
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if self._is_pd(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        identity_matrix = np.eye(A.shape[0])
        k = 1
        while not self._is_pd(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += identity_matrix * (-mineig * k ** 2 + spacing)
            k += 1

        return A3

    def _compute_ed_param_train(self, distance_func, train_distrib, classes, n_cls_i):
        """Pre-compute the training-side terms of the ED optimization problem.

        Given the class-conditional posterior clouds observed on validation
        data, this routine computes the pairwise average distances between
        classes and derives the matrices required by the quadratic program.
        These terms depend only on the validation distribution and can therefore
        be cached after ``aggregation_fit``.
        """
        n_classes = len(classes)
        K = np.zeros((n_classes, n_classes), dtype=float)
        for i in range(n_classes):
            K[i, i] = distance_func(train_distrib[classes[i]], train_distrib[classes[i]]).sum()
            for j in range(i + 1, n_classes):
                K[i, j] = distance_func(train_distrib[classes[i]], train_distrib[classes[j]]).sum()
                K[j, i] = K[i, j]

        K = K / np.dot(n_cls_i, n_cls_i.T)

        B = np.zeros((n_classes - 1, n_classes - 1), dtype=float)
        for i in range(n_classes - 1):
            B[i, i] = -K[i, i] - K[-1, -1] + 2 * K[i, -1]
            for j in range(n_classes - 1):
                if j == i:
                    continue
                B[i, j] = -K[i, j] - K[-1, -1] + K[i, -1] + K[j, -1]

        G = 2 * B
        if not self._is_pd(G):
            G = self._nearest_pd(G)

        C = -np.vstack([np.ones((1, n_classes - 1)), -np.eye(n_classes - 1)]).T
        b = -np.array([1] + [0] * (n_classes - 1), dtype=float)

        return K, G, C, b

    def _compute_ed_param_test(self, distance_func, train_distrib, test_distrib, K, classes, n_cls_i):
        """Compute the test-dependent linear term of the ED objective.

        Once the training-side matrices have been computed, each new test sample
        only requires estimating the distances between its posterior cloud and
        the class-conditional validation clouds.
        """
        n_classes = len(classes)
        Kt = np.zeros(n_classes, dtype=float)
        for i in range(n_classes):
            Kt[i] = distance_func(train_distrib[classes[i]], test_distrib).sum()

        Kt = Kt / (n_cls_i.squeeze() * float(len(test_distrib)))
        return 2 * (-Kt[:-1] + K[:-1, -1] + Kt[-1] - K[-1, -1])

    def _solve_ed(self, G, a, C, b):
        """Solve the energy-distance quadratic program.

        The optimization is carried out over the first ``n_classes - 1``
        prevalences; the prevalence of the last class is recovered afterwards by
        the simplex constraint. The resulting vector is finally normalized as a
        precaution against small numerical deviations.
        """
        quadprog = _get_quadprog()
        sol = quadprog.solve_qp(G=G, a=a, C=C, b=b)
        prevalences = sol[0]
        prevalences = np.append(prevalences, 1 - prevalences.sum())
        return F.normalize_prevalence(prevalences, method='clip')

    def aggregation_fit(self, classif_predictions, labels):
        """
        Estimate the class-conditional posterior distributions on validation
        data and pre-compute the quadratic-program parameters that depend only
        on the training side.

        In EDy, the validation posteriors are not discretized into histograms.
        Instead, each class is represented by the cloud of posterior vectors
        observed for that class, and these clouds are then compared through the
        selected pairwise distance.

        :param classif_predictions: posterior probabilities returned by the
            classifier on validation data
        :param labels: true labels associated to each posterior vector
        """
        posteriors = np.asarray(classif_predictions, dtype=float)
        labels = np.asarray(labels)

        self.train_distrib_ = {
            class_: posteriors[labels == class_] for class_ in self.classes_
        }
        self.train_n_cls_i_ = np.asarray(
            [[len(self.train_distrib_[class_])] for class_ in self.classes_],
            dtype=float,
        )

        self.K_, self.G_, self.C_, self.b_ = self._compute_ed_param_train(
            self.distance,
            self.train_distrib_,
            self.classes_,
            self.train_n_cls_i_,
        )
        return self

    def aggregate(self, posteriors: np.ndarray):
        """Estimate the prevalence vector for a test sample.

        :param posteriors: posterior probabilities returned by the classifier
            for the instances in the test sample
        :return: a prevalence vector of shape ``(n_classes,)``
        """
        posteriors = np.asarray(posteriors, dtype=float)
        self.a_ = self._compute_ed_param_test(
            self.distance,
            self.train_distrib_,
            posteriors,
            self.K_,
            self.classes_,
            self.train_n_cls_i_,
        )
        return self._solve_ed(G=self.G_, a=self.a_, C=self.C_, b=self.b_)
