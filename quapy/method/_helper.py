"""
Internal helper utilities shared by quantification methods.
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


def _get_abstention_calibrators():
    try:
        from abstention.calibration import NoBiasVectorScaling, TempScaling, VectorScaling
    except ImportError as exc:
        raise ImportError(
            "Posterior calibration for EMQ requires the optional 'abstention' package."
        ) from exc
    return {
        'nbvs': NoBiasVectorScaling(),
        'bcts': TempScaling(bias_positions='all'),
        'ts': TempScaling(),
        'vs': VectorScaling(),
    }


def _get_cvxpy():
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "RLLS requires the optional 'cvxpy' package."
        ) from exc
    return cp


def _get_quadprog():
    try:
        import quadprog
    except ImportError as exc:
        raise ImportError(
            "EDy requires the optional 'quadprog' package."
        ) from exc
    return quadprog


def _labels_to_indices(labels, classes):
    encoder = LabelEncoder().fit(classes)
    return encoder.transform(labels)


def _rlls_check_mode(mode):
    valid = {'soft', 'hard'}
    if mode not in valid:
        raise ValueError(f'unknown mode {mode!r}; valid ones are {valid}')
    return mode


def _rlls_joint_distribution(posteriors, labels, classes, mode='soft'):
    mode = _rlls_check_mode(mode)
    posteriors = np.asarray(posteriors, dtype=float)
    labels = np.asarray(labels)
    n_samples, n_classes = posteriors.shape
    assert n_classes == len(classes), 'wrong number of posterior columns'

    if mode == 'hard':
        pred = np.argmax(posteriors, axis=1)
        encoded_labels = _labels_to_indices(labels, classes)
        joint = confusion_matrix(encoded_labels, pred, labels=np.arange(n_classes)).T.astype(float)
        return joint / n_samples

    joint = np.zeros((n_classes, n_classes), dtype=float)
    for class_index, class_ in enumerate(classes):
        idx = labels == class_
        if idx.any():
            joint[:, class_index] = posteriors[idx].sum(axis=0)
    return joint / n_samples


def _rlls_predicted_marginal(posteriors, mode='soft'):
    mode = _rlls_check_mode(mode)
    posteriors = np.asarray(posteriors, dtype=float)
    if mode == 'soft':
        return posteriors.mean(axis=0)

    pred = np.argmax(posteriors, axis=1)
    counts = np.bincount(pred, minlength=posteriors.shape[1]).astype(float)
    return counts / counts.sum()


def _rlls_compute_3deltaC(n_classes, n_train, delta):
    return 3 * (
        2 * np.log(2 * n_classes / delta) / (3 * n_train)
        + np.sqrt(2 * np.log(2 * n_classes / delta) / n_train)
    )


def _rlls_compute_weights(C_zy, qz, pz, rho, clip=False):
    cp = _get_cvxpy()

    n_classes = C_zy.shape[0]
    theta = cp.Variable(n_classes)
    b = qz - pz
    objective = cp.Minimize(cp.pnorm(C_zy @ theta - b) + rho * cp.pnorm(theta))
    constraints = [-1 <= theta]
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(verbose=False, solver=cp.SCS)
    except cp.error.SolverError:
        problem.solve(verbose=False, solver=cp.SCS, use_indirect=True)

    if theta.value is None:
        raise RuntimeError('RLLS optimization failed to produce a solution')

    w = 1 + np.asarray(theta.value, dtype=float)
    if clip and np.any(w < 0):
        w = np.clip(w, 0, None)
    return w
