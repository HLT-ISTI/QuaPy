from typing import Union
import numpy as np
from sklearn.base import BaseEstimator
from quapy.data import LabelledCollection
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from quapy.method.aggregative import AggregativeProbabilisticQuantifier, cross_generate_predictions, _get_divergence
import quadprog


class EDy(AggregativeProbabilisticQuantifier):
    """This class and its methods have been imported from quantificationlib <https://github.com/AICGijon/quantificationlib>"""

    def __init__(self, classifier: BaseEstimator, val_split=10, distance=manhattan_distances, n_jobs=None, random_state=0):
        
        self.classifier = classifier
        self.val_split = val_split
        self.distance = distance
        self.n_jobs = n_jobs
        self.random_state=random_state
        self.train_n_cls_i_ = None
        self.train_distrib_ = None
        self.K_ = None
        #  variables for solving the optimization problem
        self.G_ = None
        self.C_ = None
        self.b_ = None
        self.a_ = None

    def _is_pd(self, m):
        """ Checks whether a matrix is positive definite or not

            It is based on dpofa function, a version of the dpofa function included in quadprog library. When dpofa
            returns 0 the matrix is positive definite.

            Parameters
            ----------
            m : symmetric matrix, typically the shape is (n_classes, n_classes)
                The matrix to check whether it is positive definite or not

            Returns
            -------
            A boolean, True when m is positive definite and False otherwise

        """
        return self._dpofa(m)[0] == 0

    def _dpofa(self, m):
        """ Factors a symmetric positive definite matrix

            This is a version of the dpofa function included in quadprog library. Here, it is mainly used to check
            whether a matrix is positive definite or not

            Parameters
            ----------
            m : symmetric matrix, typically the shape is (n_classes, n_classes)
                The matrix to be factored. Only the diagonal and upper triangle are used

            Returns
            -------
            k : int, 
                This value is:

                == 0  if m is positive definite and the factorization has been completed  \n
                >  0  when the leading minor of order k is not positive definite

            r : array, an upper triangular matrix
                When k==0, the factorization is complete and `r.T.dot(r) == m`.
                The strict lower triangle is unaltered (it is equal to the strict lower triangle of matrix m), so it
                could be different from 0.
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
                return k+1, r
            r[k, k] = np.sqrt(s)
        return 0, r
    
    def _nearest_pd(self, A):
        """ Find the nearest positive-definite matrix to input

            A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which credits [2].

            References
            ----------
            [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

            [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988):
                https://doi.org/10.1016/0024-3795(88)90223-6
        """
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if self._is_pd(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        indendity_matrix = np.eye(A.shape[0])
        k = 1
        while not self._is_pd(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += indendity_matrix * (-mineig * k ** 2 + spacing)
            k += 1

        return A3


    def _compute_ed_param_train(self, distance_func, train_distrib, classes, n_cls_i):
        """ Computes params related to the train distribution for solving ED-problems using `quadprog.solve_qp`

            Parameters
            ----------
            distance_func : function
                The function used to measure the distance between each pair of examples

            train_distrib : array, shape (n_bins * n_classes, n_classes)
                Represents the distribution of each class in the training set

            classes : ndarray, shape (n_classes, )
                Class labels

            n_cls_i: ndarray, shape (n_classes, )
                The number of examples of each class

            Returns
            -------
            K : array, shape (n_classes, n_classes)
                Average distance between each pair of classes in the training set

            G : array, shape (n_classes - 1, n_classes - 1)

            C : array, shape (n_classes - 1, n_constraints)
                n_constraints will be equal to the number of classes (n_classes)

            b : array, shape (n_constraints,)

            References
            ----------
            Alberto Castaño, Laura Morán-Fernández, Jaime Alonso, Verónica Bolón-Canedo, Amparo Alonso-Betanzos,
            Juan José del Coz: An analysis of quantification methods based on matching distributions

            Hideko Kawakubo, Marthinus Christoffel Du Plessis, and Masashi Sugiyama. 2016. Computationally efficient
            class-prior estimation under class balance change using energy distance. Transactions on Information
            and Systems 99, 1 (2016), 176–186.
        """
        n_classes = len(classes)
        #  computing sum de distances for each pair of classes
        K = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            K[i, i] = distance_func(train_distrib[classes[i]], train_distrib[classes[i]]).sum()
            for j in range(i + 1, n_classes):
                K[i, j] = distance_func(train_distrib[classes[i]], train_distrib[classes[j]]).sum()
                K[j, i] = K[i, j]

        #  average distance
        K = K / np.dot(n_cls_i, n_cls_i.T)

        B = np.zeros((n_classes - 1, n_classes - 1))
        for i in range(n_classes - 1):
            B[i, i] = - K[i, i] - K[-1, -1] + 2 * K[i, -1]
            for j in range(n_classes - 1):
                if j == i:
                    continue
                B[i, j] = - K[i, j] - K[-1, -1] + K[i, -1] + K[j, -1]

        #  computing the terms for the optimization problem
        G = 2 * B
        if not self._is_pd(G):
            G = self._nearest_pd(G)

        C = -np.vstack([np.ones((1, n_classes - 1)), -np.eye(n_classes - 1)]).T
        b = -np.array([1] + [0] * (n_classes - 1), dtype=float)

        return K, G, C, b
    
    def _compute_ed_param_test(self, distance_func, train_distrib, test_distrib, K, classes, n_cls_i):
        """ Computes params related to the test distribution for solving ED-problems using `quadprog.solve_qp`

            Parameters
            ----------
            distance_func : function
                The function used to measure the distance between each pair of examples

            train_distrib : array, shape (n_bins * n_classes, n_classes)
                Represents the distribution of each class in the training set

            test_distrib : array, shape (n_bins * n_classes, 1)
                Represents the distribution of the testing set

            K : array, shape (n_classes, n_classes)
                Average distance between each pair of classes in the training set

            classes : ndarray, shape (n_classes, )
                Class labels

            n_cls_i: ndarray, shape (n_classes, )
                The number of examples of each class

            Returns
            -------
            a : array, shape (n_classes, )
                Term a for solving optimization problems using `quadprog.solve_qp`

    
            References
            ----------
            Alberto Castaño, Laura Morán-Fernández, Jaime Alonso, Verónica Bolón-Canedo, Amparo Alonso-Betanzos,
            Juan José del Coz: An analysis of quantification methods based on matching distributions

            Hideko Kawakubo, Marthinus Christoffel Du Plessis, and Masashi Sugiyama. 2016. Computationally efficient
            class-prior estimation under class balance change using energy distance. Transactions on Information
            and Systems 99, 1 (2016), 176–186.
        """
        n_classes = len(classes)
        Kt = np.zeros(n_classes)
        for i in range(n_classes):
            Kt[i] = distance_func(train_distrib[classes[i]], test_distrib).sum()

        Kt = Kt / (n_cls_i.squeeze() * float(len(test_distrib)))

        a = 2 * (- Kt[:-1] + K[:-1, -1] + Kt[-1] - K[-1, -1])
        return a
    
    def _solve_ed(self, G, a, C, b):
        """ Solves the optimization problem for ED-based quantifiers

            It just calls `quadprog.solve_qp` with the appropriate parameters. These paremeters were computed
            before by calling `compute_ed_param_train` and `compute_ed_param_test`.
            In the derivation of the optimization problem, the last class is put in terms of the rest of classes. Thus,
            we have to add 1-prevalences.sum() which it is the prevalence of the last class

            Parameters
            ----------
            G : array, shape (n_classes, n_classes)

            C : array, shape (n_classes, n_constraints)
                n_constraints will be n_classes + 1

            b : array, shape (n_constraints,)

            a : array, shape (n_classes, )

            Returns
            -------
            prevalences : array, shape=(n_classes, )
            Vector containing the predicted prevalence for each class

            Notes
            -----   
            G, C and b are computed by `compute_ed_param_train` and a by `compute_ed_param_test`
            
            References
            ----------
            Alberto Castaño, Laura Morán-Fernández, Jaime Alonso, Verónica Bolón-Canedo, Amparo Alonso-Betanzos,
            Juan José del Coz: An analysis of quantification methods based on matching distributions

            Hideko Kawakubo, Marthinus Christoffel Du Plessis, and Masashi Sugiyama. 2016. Computationally efficient
            class-prior estimation under class balance change using energy distance. Transactions on Information
            and Systems 99, 1 (2016), 176–186.
        """
        sol = quadprog.solve_qp(G=G, a=a, C=C, b=b)
        prevalences = sol[0]
        # the last class was removed from the problem, its prevalence is 1 - the sum of prevalences for the other classes
        return np.append(prevalences, 1 - prevalences.sum())

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split: Union[float, LabelledCollection] = None):
        if val_split is None:
            val_split = self.val_split

        self.classifier, y, posteriors, _, _ = cross_generate_predictions(
            data, self.classifier, val_split, probabilistic=True, fit_classifier=fit_classifier, n_jobs=self.n_jobs
        )

        n_classes = len(self.classifier.classes_)

        self.train_distrib_ = dict.fromkeys(self.classifier.classes_)
        self.train_n_cls_i_ = np.zeros((n_classes, 1))
        for n_cls, cls in enumerate(self.classifier.classes_):
            self.train_distrib_[cls] = posteriors[y == cls]
            self.train_n_cls_i_[n_cls, 0] = len(self.train_distrib_[cls])

        self.K_, self.G_, self.C_, self.b_ = self._compute_ed_param_train(self.distance, self.train_distrib_,
                                                                    self.classes_, self.train_n_cls_i_)


        return self

    def aggregate(self, posteriors: np.ndarray):
        self.a_ = self._compute_ed_param_test(self.distance, self.train_distrib_, posteriors, self.K_,
                                        self.classes_, self.train_n_cls_i_)

        prevalences = self._solve_ed(G=self.G_, a=self.a_, C=self.C_, b=self.b_)

        return prevalences