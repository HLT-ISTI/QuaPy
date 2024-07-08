import itertools
from collections import defaultdict
import scipy
import numpy as np
from typing import Literal


def prevalence_linspace(n_prevalences=21, repeats=1, smooth_limits_epsilon=0.01):
    """
    Produces an array of uniformly separated values of prevalence.
    By default, produces an array of 21 prevalence values, with
    step 0.05 and with the limits smoothed, i.e.:
    [0.01, 0.05, 0.10, 0.15, ..., 0.90, 0.95, 0.99]

    :param n_prevalences: the number of prevalence values to sample from the [0,1] interval (default 21)
    :param repeats: number of times each prevalence is to be repeated (defaults to 1)
    :param smooth_limits_epsilon: the quantity to add and subtract to the limits 0 and 1
    :return: an array of uniformly separated prevalence values
    """
    p = np.linspace(0., 1., num=n_prevalences, endpoint=True)
    p[0] += smooth_limits_epsilon
    p[-1] -= smooth_limits_epsilon
    if p[0] > p[1]:
        raise ValueError(f'the smoothing in the limits is greater than the prevalence step')
    if repeats > 1:
        p = np.repeat(p, repeats)
    return p


def prevalence_from_labels(labels, classes):
    """
    Computed the prevalence values from a vector of labels.

    :param labels: array-like of shape `(n_instances)` with the label for each instance
    :param classes: the class labels. This is needed in order to correctly compute the prevalence vector even when
        some classes have no examples.
    :return: an ndarray of shape `(len(classes))` with the class prevalence values
    """
    if labels.ndim != 1:
        raise ValueError(f'param labels does not seem to be a ndarray of label predictions')
    unique, counts = np.unique(labels, return_counts=True)
    by_class = defaultdict(lambda:0, dict(zip(unique, counts)))
    prevalences = np.asarray([by_class[class_] for class_ in classes], dtype=float)
    prevalences /= prevalences.sum()
    return prevalences


def prevalence_from_probabilities(posteriors, binarize: bool = False):
    """
    Returns a vector of prevalence values from a matrix of posterior probabilities.

    :param posteriors: array-like of shape `(n_instances, n_classes,)` with posterior probabilities for each class
    :param binarize: set to True (default is False) for computing the prevalence values on crisp decisions (i.e.,
        converting the vectors of posterior probabilities into class indices, by taking the argmax).
    :return: array of shape `(n_classes,)` containing the prevalence values
    """
    if posteriors.ndim != 2:
        raise ValueError(f'param posteriors does not seem to be a ndarray of posteior probabilities')
    if binarize:
        predictions = np.argmax(posteriors, axis=-1)
        return prevalence_from_labels(predictions, np.arange(posteriors.shape[1]))
    else:
        prevalences = posteriors.mean(axis=0)
        prevalences /= prevalences.sum()
        return prevalences


def HellingerDistance(P, Q):
    """
    Computes the Hellingher Distance (HD) between (discretized) distributions `P` and `Q`.
    The HD for two discrete distributions of `k` bins is defined as:

    .. math::
        HD(P,Q) = \\frac{ 1 }{ \\sqrt{ 2 } } \\sqrt{ \\sum_{i=1}^k ( \\sqrt{p_i} - \\sqrt{q_i} )^2 }

    :param P: real-valued array-like of shape `(k,)` representing a discrete distribution
    :param Q: real-valued array-like of shape `(k,)` representing a discrete distribution
    :return: float
    """
    return np.sqrt(np.sum((np.sqrt(P) - np.sqrt(Q))**2))


def TopsoeDistance(P, Q, epsilon=1e-20):
    """
    Topsoe distance between two (discretized) distributions `P` and `Q`.
    The Topsoe distance for two discrete distributions of `k` bins is defined as:

    .. math::
        Topsoe(P,Q) = \\sum_{i=1}^k \\left( p_i \\log\\left(\\frac{ 2 p_i + \\epsilon }{ p_i+q_i+\\epsilon }\\right) +
            q_i \\log\\left(\\frac{ 2 q_i + \\epsilon }{ p_i+q_i+\\epsilon }\\right) \\right)

    :param P: real-valued array-like of shape `(k,)` representing a discrete distribution
    :param Q: real-valued array-like of shape `(k,)` representing a discrete distribution
    :param epsilon: small value to smooth the distributions for numerical stability
    :return: float
    """
    return np.sum(P*np.log((2*P+epsilon)/(P+Q+epsilon)) + Q*np.log((2*Q+epsilon)/(P+Q+epsilon)))
                  

def CauchySchwarz(P, Q, epsilon=1e-20):
    """
    Cauchy-Schwarz divergence between two (discretized) distributions `P` and `Q`.
    The Cauchy-Schwarz divergence for two discrete distributions of `k` bins is defined as:

    .. math::
        CS(P,Q) = \\frac{ \\sum_{i=1}^k  p_i q_i }{
            \\left( \\sum_{i=1}^k  p^2_i \\right) \\left( \\sum_{i=1}^k  q^2_i \\right)
        }

    :param P: real-valued array-like of shape `(k,)` representing a discrete distribution
    :param Q: real-valued array-like of shape `(k,)` representing a discrete distribution
    :param epsilon: small value to smooth the distributions for numerical stability
    :return: float
    """
    P += epsilon
    Q += epsilon
    return - np.log(sum(P * Q) / np.sqrt(sum(P ** 2) * sum(Q ** 2)))


def uniform_prevalence_sampling(n_classes, size=1):
    """
    Implements the `Kraemer algorithm <http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf>`_
    for sampling uniformly at random from the unit simplex. This implementation is adapted from this
    `post <https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex>_`.

    :param n_classes: integer, number of classes (dimensionality of the simplex)
    :param size: number of samples to return
    :return: `np.ndarray` of shape `(size, n_classes,)` if `size>1`, or of shape `(n_classes,)` otherwise
    """
    if n_classes == 2:
        u = np.random.rand(size)
        u = np.vstack([1-u, u]).T
    else:
        u = np.random.rand(size, n_classes-1)
        u.sort(axis=-1)
        _0s = np.zeros(shape=(size, 1))
        _1s = np.ones(shape=(size, 1))
        a = np.hstack([_0s, u])
        b = np.hstack([u, _1s])
        u = b-a
    if size == 1:
        u = u.flatten()
    return u


uniform_simplex_sampling = uniform_prevalence_sampling


def strprev(prevalences, prec=3):
    """
    Returns a string representation for a prevalence vector. E.g.,

    >>> strprev([1/3, 2/3], prec=2)
    >>> '[0.33, 0.67]'

    :param prevalences: a vector of prevalence values
    :param prec: float precision
    :return: string
    """
    return '['+ ', '.join([f'{p:.{prec}f}' for p in prevalences]) + ']'


def adjusted_quantification(prevalence_estim, tpr, fpr, clip=True):
    """
    Implements the adjustment of ACC and PACC for the binary case. The adjustment for a prevalence estimate of the
    positive class `p` comes down to computing:

    .. math::
        ACC(p) = \\frac{ p - fpr }{ tpr - fpr }

    :param prevalence_estim: float, the estimated value for the positive class
    :param tpr: float, the true positive rate of the classifier
    :param fpr: float, the false positive rate of the classifier
    :param clip: set to True (default) to clip values that might exceed the range [0,1]
    :return: float, the adjusted count
    """

    den = tpr - fpr
    if den == 0:
        den += 1e-8
    adjusted = (prevalence_estim - fpr) / den
    if clip:
        adjusted = np.clip(adjusted, 0., 1.)
    return adjusted


def normalize_prevalence(prevalences, method='clip'):
    """
    Normalizes a vector or matrix of prevalence values. The normalization consists of applying a L1 normalization in
    cases in which the prevalence values are not all-zeros, and to convert the prevalence values into `1/n_classes` in
    cases in which all values are zero.

    :param prevalences: array-like of shape `(n_classes,)` or of shape `(n_samples, n_classes,)` with prevalence values
    :param str method: indicates the normalization method to employ, options are:

        * `l1`: applies L1 normalization (default); a 0 vector is mapped onto the uniform prevalence
        * `clip`: clip values in [0,1] and then rescales so that the L1 norm is 1
        * `mapsimplex`: projects vectors onto the probability simplex. This implementation relies on
          `Mathieu Blondel's projection_simplex_sort <https://gist.github.com/mblondel/6f3b7aaad90606b98f71>`_
        * `softmax`: applies softmax to all vectors
        * `condsoftmax`: applies softmax only to invalid prevalence vectors

    :return: a normalized vector or matrix of prevalence values
    """
    if method in ['none', None]:
        return prevalences

    prevalences = np.asarray(prevalences, dtype=float)

    if method=='clip':
        normalized = clip(prevalences)  # no need to check afterwards
    else:
        raise ValueError(f'unknown {method=}, valid ones are ["l1", "clip", "mapsimplex", "softmax", "condsoftmax"]')

    return normalized



def clip(prevalences) -> np.ndarray:
    """
    Clips the values in [0,1] and then applies the L1 normalization.

    :param prevalences: array-like of shape `(n_classes,)` or of shape `(n_samples, n_classes,)` with prevalence values
    :return: np.ndarray representing a valid distribution
    """
    clipped = np.clip(prevalences, 0, 1)
    normalized = l1_norm(clipped)
    return normalized

def l1_norm(prevalences) -> np.ndarray:
    """
    Applies L1 normalization to the `unnormalized_arr` so that it becomes a valid prevalence
    vector. Zero vectors are mapped onto the uniform distribution. Raises an exception if
    the resulting vectors are not valid distributions. This may happen when the original
    prevalence vectors contain negative values. Use the `clip` normalization function
    instead to avoid this possibility.

    :param prevalences: array-like of shape `(n_classes,)` or of shape `(n_samples, n_classes,)` with prevalence values
    :return: np.ndarray representing a valid distribution
    """
    n_classes = prevalences.shape[-1]
    accum = prevalences.sum(axis=-1, keepdims=True)
    prevalences = np.true_divide(prevalences, accum, where=accum > 0)
    allzeros = accum.flatten() == 0
    if any(allzeros):
        if prevalences.ndim == 1:
            prevalences = np.full(shape=n_classes, fill_value=1. / n_classes)
        else:
            prevalences[allzeros] = np.full(shape=n_classes, fill_value=1. / n_classes)
    return prevalences

def __num_prevalence_combinations_depr(n_prevpoints:int, n_classes:int, n_repeats:int=1):
    """
    Computes the number of prevalence combinations in the n_classes-dimensional simplex if `nprevpoints` equally distant
    prevalence values are generated and `n_repeats` repetitions are requested.

    :param n_classes: integer, number of classes
    :param n_prevpoints: integer, number of prevalence points.
    :param n_repeats: integer, number of repetitions for each prevalence combination
    :return: The number of possible combinations. For example, if n_classes=2, n_prevpoints=5, n_repeats=1, then the
        number of possible combinations are 5, i.e.: [0,1], [0.25,0.75], [0.50,0.50], [0.75,0.25], and [1.0,0.0]
    """
    __cache={}
    def __f(nc,np):
        if (nc,np) in __cache:  # cached result
            return __cache[(nc,np)]
        if nc==1:  # stop condition
            return 1
        else:  # recursive call
            x = sum([__f(nc-1, np-i) for i in range(np)])
            __cache[(nc,np)] = x
            return x
    return __f(n_classes, n_prevpoints) * n_repeats


def num_prevalence_combinations(n_prevpoints:int, n_classes:int, n_repeats:int=1):
    """
    Computes the number of valid prevalence combinations in the n_classes-dimensional simplex if `n_prevpoints` equally
    distant prevalence values are generated and `n_repeats` repetitions are requested.
    The computation comes down to calculating:

    .. math::
        \\binom{N+C-1}{C-1} \\times r

    where `N` is `n_prevpoints-1`, i.e., the number of probability mass blocks to allocate, `C` is the number of
    classes, and `r` is `n_repeats`. This solution comes from the
    `Stars and Bars <https://brilliant.org/wiki/integer-equations-star-and-bars/>`_ problem.

    :param n_classes: integer, number of classes
    :param n_prevpoints: integer, number of prevalence points.
    :param n_repeats: integer, number of repetitions for each prevalence combination
    :return: The number of possible combinations. For example, if n_classes=2, n_prevpoints=5, n_repeats=1, then the
        number of possible combinations are 5, i.e.: [0,1], [0.25,0.75], [0.50,0.50], [0.75,0.25], and [1.0,0.0]
    """
    N = n_prevpoints-1
    C = n_classes
    r = n_repeats
    return int(scipy.special.binom(N + C - 1, C - 1) * r)


def get_nprevpoints_approximation(combinations_budget:int, n_classes:int, n_repeats:int=1):
    """
    Searches for the largest number of (equidistant) prevalence points to define for each of the `n_classes` classes so
    that the number of valid prevalence values generated as combinations of prevalence points (points in a
    `n_classes`-dimensional simplex) do not exceed combinations_budget.

    :param combinations_budget: integer, maximum number of combinations allowed
    :param n_classes: integer, number of classes
    :param n_repeats: integer, number of repetitions for each prevalence combination
    :return: the largest number of prevalence points that generate less than combinations_budget valid prevalences
    """
    assert n_classes > 0 and n_repeats > 0 and combinations_budget > 0, 'parameters must be positive integers'
    n_prevpoints = 1
    while True:
        combinations = num_prevalence_combinations(n_prevpoints, n_classes, n_repeats)
        if combinations > combinations_budget:
            return n_prevpoints-1
        else:
            n_prevpoints += 1


def check_prevalence_vector(p, raise_exception=False, toleranze=1e-08):
    """
    Checks that p is a valid prevalence vector, i.e., that it contains values in [0,1] and that the values sum up to 1.

    :param p: the prevalence vector to check
    :return: True if `p` is valid, False otherwise
    """
    p = np.asarray(p)
    if not all(p>=0):
        if raise_exception:
            raise ValueError('the prevalence vector contains negative numbers')
        return False
    if not all(p<=1):
        if raise_exception:
            raise ValueError('the prevalence vector contains values >1')
        return False
    if not np.isclose(p.sum(), 1, atol=toleranze):
        if raise_exception:
            raise ValueError('the prevalence vector does not sum up to 1')
        return False
    return True


def optim_minimize(loss, n_classes):
    """
    Searches for the optimal prevalence values, i.e., an `n_classes`-dimensional vector of the (`n_classes`-1)-simplex
    that yields the smallest lost. This optimization is carried out by means of a constrained search using scipy's
    SLSQP routine.

    :param loss: (callable) the function to minimize
    :param n_classes: (int) the number of classes, i.e., the dimensionality of the prevalence vector
    :return: (ndarray) the best prevalence vector found
    """
    from scipy import optimize

    # the initial point is set as the uniform distribution
    uniform_distribution = np.full(fill_value=1 / n_classes, shape=(n_classes,))

    # solutions are bounded to those contained in the unit-simplex
    bounds = tuple((0, 1) for _ in range(n_classes))  # values in [0,1]
    constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})  # values summing up to 1
    r = optimize.minimize(loss, x0=uniform_distribution, method='SLSQP', bounds=bounds, constraints=constraints)
    return r.x


def solve_adjustment(
    class_conditional_rates: np.ndarray,
    unadjusted_counts: np.ndarray,
    method: Literal["inversion", "invariant-ratio"],
    solver: Literal["exact", "minimize", "exact-raise", "exact-cc"]) -> np.ndarray:
    """
    Function that tries to solve for :math:`p` the equation :math:`q = M p`, where :math:`q` is the vector of
    `unadjusted counts` (as estimated, e.g., via classify and count) with :math:`q_i` an estimate of
    :math:`P(\hat{Y}=y_i)`, and where :math:`M` is the matrix of `class-conditional rates` with :math:`M_{ij}` an
    estimate of :math:`P(\hat{Y}=y_i|Y=y_j)`.

    :param class_conditional_rates: array of shape `(n_classes, n_classes,)` with entry `(i,j)` being the estimate
        of :math:`P(\hat{Y}=y_i|Y=y_j)`, that is, the probability that an instance that belongs to class :math:`y_j`
        ends up being classified as belonging to class :math:`y_i`

    :param unadjusted_counts: array of shape `(n_classes,)` containing the unadjusted prevalence values (e.g., as
        estimated by CC or PCC)

    :param str method: indicates the adjustment method to be used. Valid options are:

        * `inversion`: tries to solve the equation :math:`q = M p` as :math:`p = M^{-1} q` where
          :math:`M^{-1}` is the matrix inversion of :math:`M`. This inversion may not exist in
          degenerated cases.
        * `invariant-ratio`: invariant ratio estimator of `Vaz et al. 2018 <https://jmlr.org/papers/v20/18-456.html>`_,
          which replaces the last equation in :math:`M` with the normalization condition (i.e., that the sum of
          all prevalence values must equal 1).

    :param str solver: the method to use for solving the system of linear equations. Valid options are:

        * `exact-raise`: tries to solve the system using matrix inversion. Raises an error if the matrix has rank
          strictly lower than `n_classes`.
        * `exact-cc`: if the matrix is not full rank, returns :math:`q` (i.e., the unadjusted counts) as the estimates
        * `exact`: deprecated, defaults to 'exact-cc' (will be removed in future versions)
        * `minimize`: minimizes a loss, so the solution always exists
    """
    if solver == "exact":
        print("The 'exact' solver is deprecated. Use 'exact-raise' or 'exact-cc'")
        solver = "exact-cc"

    A = np.asarray(class_conditional_rates, dtype=float)
    B = np.asarray(unadjusted_counts, dtype=float)

    if method == "inversion":
        pass  # We leave A and B unchanged
    elif method == "invariant-ratio":
        # Change the last equation to replace it with the normalization condition
        A[-1, :] = 1.0
        B[-1] = 1.0
    else:
        raise ValueError(f"unknown {method=}")

    if solver == "minimize":
        def loss(prev):
            return np.linalg.norm(A @ prev - B)
        return optim_minimize(loss, n_classes=A.shape[0])
    elif solver in ["exact-raise", "exact-cc"]:
        # Solvers based on matrix inversion, so we use try/except block
        try:
            return np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            # The matrix is not invertible.
            # Depending on the solver, we either raise an error
            # or return the classifier predictions without adjustment
            if solver == "exact-raise":
                raise
            elif solver == "exact-cc":
                return unadjusted_counts
            else:
                raise ValueError(f"Solver {solver} not known.")
    else:
        raise ValueError(f'unknown {solver=}')


def as_binary_prevalence(positive_prevalence, clip_if_necessary=False):
    """
    Helper that, given a float representing the prevalence for the positive class, returns a np.ndarray of two
    values representing a binary distribution.

    :param positive_prevalence: prevalence for the positive class
    :param clip_if_necessary: if True, clips the value in [0,1] in order to guarantee the resulting distribution
        is valid. If False, it then checks that the value is in the valid range, and raises an error if not.
    :return: np.ndarray of shape `(2,)`
    """
    if clip_if_necessary:
        positive_prevalence = np.clip(positive_prevalence, 0, 1)
    else:
        assert 0 <= positive_prevalence <= 1, 'the value provided is not a valid prevalence for the positive class'
    return np.asarray([1-positive_prevalence, positive_prevalence]).T