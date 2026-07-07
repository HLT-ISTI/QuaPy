from typing import Callable, Union

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

import quapy as qp
import quapy.functional as F
from quapy.method._helper import _get_quadprog


class _EnergyDistanceCore:
    """Shared numerical core for energy-distance quantifiers."""

    def _check_ed_init_parameters(self):
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
        """Check whether a symmetric matrix is positive definite."""
        return self._dpofa(m)[0] == 0

    def _dpofa(self, m):
        """Factor a symmetric positive definite matrix."""
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
        """Project a matrix onto the cone of positive-definite matrices."""
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

    def _compute_ed_param_train(self, distance_func, train_distrib, n_cls_i):
        """Pre-compute the training-side terms of the ED optimization problem."""
        n_classes = len(train_distrib)
        K = np.zeros((n_classes, n_classes), dtype=float)
        for i in range(n_classes):
            K[i, i] = distance_func(train_distrib[i], train_distrib[i]).sum()
            for j in range(i + 1, n_classes):
                K[i, j] = distance_func(train_distrib[i], train_distrib[j]).sum()
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

    def _compute_ed_param_test(self, distance_func, train_distrib, test_distrib, K, n_cls_i):
        """Compute the test-dependent linear term of the ED objective."""
        n_classes = len(train_distrib)
        Kt = np.zeros(n_classes, dtype=float)
        for i in range(n_classes):
            Kt[i] = distance_func(train_distrib[i], test_distrib).sum()

        Kt = Kt / (n_cls_i.squeeze() * float(test_distrib.shape[0]))
        return 2 * (-Kt[:-1] + K[:-1, -1] + Kt[-1] - K[-1, -1])

    def _solve_ed(self, G, a, C, b):
        """Solve the energy-distance quadratic program."""
        quadprog = _get_quadprog()
        sol = quadprog.solve_qp(G=G, a=a, C=C, b=b)
        prevalences = sol[0]
        prevalences = np.append(prevalences, 1 - prevalences.sum())
        return F.normalize_prevalence(prevalences, method='clip')

    def _fit_energy_model(self, train_distrib):
        self.train_distrib_ = tuple(train_distrib)
        self.train_n_cls_i_ = np.asarray(
            [[distrib.shape[0]] for distrib in self.train_distrib_],
            dtype=float,
        )
        self.K_, self.G_, self.C_, self.b_ = self._compute_ed_param_train(
            self.distance,
            self.train_distrib_,
            self.train_n_cls_i_,
        )
        return self

    def _predict_energy(self, test_distrib):
        self.a_ = self._compute_ed_param_test(
            self.distance,
            self.train_distrib_,
            test_distrib,
            self.K_,
            self.train_n_cls_i_,
        )
        return self._solve_ed(G=self.G_, a=self.a_, C=self.C_, b=self.b_)
