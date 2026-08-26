import numpy as np


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    v = np.maximum(v, eps)
    return v / v.sum()


def bayes_prior_update(probs, new_prior, source_prior, eps=1e-12):
    """
    Applies the standard label-shift prior correction:

        p_new(y|x) ∝ p_old(y|x) * new_prior(y) / source_prior(y)

    Parameters
    ----------
    probs : array, shape (n_samples, n_classes) or (n_classes,)
        Probabilistic outputs of the classifier.
    new_prior : array, shape (n_classes,)
        Current estimate of the target prior.
    source_prior : array, shape (n_classes,)
        Source/training prior.
    """
    probs = np.asarray(probs, dtype=float)
    one_dim = probs.ndim == 1
    if one_dim:
        probs = probs[None, :]

    new_prior = normalize(new_prior, eps=eps)
    source_prior = normalize(source_prior, eps=eps)

    weights = new_prior / np.maximum(source_prior, eps)
    updated = probs * weights[None, :]
    updated = updated / np.maximum(updated.sum(axis=1, keepdims=True), eps)

    return updated[0] if one_dim else updated


def confusion_statistic(y_true, y_pred, n_classes, mode="recall", eps=1e-12):
    """
    Computes the minimum diagonal statistic used to choose tau.

    mode="recall":
        diag(C) / row sums, i.e. per-true-class recall.

    mode="precision":
        diag(C) / column sums, i.e. per-predicted-class precision.

    The paper calls this quantity 'minimum recall', but also describes
    a column-normalized confusion matrix, which would correspond more
    closely to precision. I expose both options.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    C = np.zeros((n_classes, n_classes), dtype=float)
    for yt, yp in zip(y_true, y_pred):
        C[yt, yp] += 1.0

    if mode == "recall":
        denom = C.sum(axis=1)
    elif mode == "precision":
        denom = C.sum(axis=0)
    else:
        raise ValueError("mode must be 'recall' or 'precision'.")

    diag = np.diag(C)
    valid = denom > 0

    if not np.any(valid):
        raise ValueError("No valid classes found in the confusion matrix.")

    scores = diag[valid] / np.maximum(denom[valid], eps)
    return float(np.min(scores))


def choose_tau_from_validation(
    target_probs,
    val_probs,
    y_val,
    mode="recall",
    retain_fraction=None,
):
    """
    Chooses tau following the spirit of Section 4.1.

    The paper says tau is selected from the top n percentile of target
    confidences, with n = min-recall * 100. To make the selected set A
    contain approximately min-recall fraction of the target samples, we set:

        tau = quantile(max_probs, 1 - min_recall)

    so that about min_recall of the target points satisfy max_prob >= tau.

    If retain_fraction is provided, it overrides the validation-derived value.
    """
    target_probs = np.asarray(target_probs, dtype=float)
    val_probs = np.asarray(val_probs, dtype=float)
    y_val = np.asarray(y_val, dtype=int)

    n_classes = target_probs.shape[1]
    y_val_pred = val_probs.argmax(axis=1)

    if retain_fraction is None:
        retain_fraction = confusion_statistic(
            y_true=y_val,
            y_pred=y_val_pred,
            n_classes=n_classes,
            mode=mode,
        )

    retain_fraction = float(np.clip(retain_fraction, 0.0, 1.0))

    target_conf = target_probs.max(axis=1)

    if retain_fraction <= 0:
        tau = np.inf
    elif retain_fraction >= 1:
        tau = -np.inf
    else:
        tau = np.quantile(target_conf, 1.0 - retain_fraction)

    return tau, retain_fraction


def leip(
    target_probs,
    source_prior,
    tau=None,
    val_probs=None,
    y_val=None,
    threshold_mode="recall",
    count_smoothing=0.0,
    eps=1e-12,
    return_details=False,
):
    """
    LEIP: Label shift Estimation with Incremental Prior update.

    Parameters
    ----------
    target_probs : array, shape (n_target, n_classes)
        Probabilistic classifier outputs on the target/test set.
    source_prior : array, shape (n_classes,)
        Source class prior p_s(y).
    tau : float or None
        Confidence threshold. If None, it is estimated using validation data.
    val_probs : array, shape (n_val, n_classes), optional
        Validation probabilistic outputs, required if tau is None.
    y_val : array, shape (n_val,), optional
        Validation labels, required if tau is None.
    threshold_mode : {"recall", "precision"}
        Statistic used to choose tau from validation data.
    count_smoothing : float
        Optional additive smoothing for pseudo-label counts. Set to 0.0
        for a closer implementation of the paper; use a small value such
        as 1e-8 for extra numerical robustness.
    eps : float
        Numerical stabilizer.
    return_details : bool
        If True, returns diagnostic information.

    Returns
    -------
    estimated_prior : array, shape (n_classes,)
        Estimated target class distribution.
    details : dict, optional
        Returned only if return_details=True.
    """
    target_probs = np.asarray(target_probs, dtype=float)
    if target_probs.ndim != 2:
        raise ValueError("target_probs must have shape (n_samples, n_classes).")

    n_target, n_classes = target_probs.shape
    source_prior = normalize(source_prior, eps=eps)

    if n_classes != len(source_prior):
        raise ValueError("source_prior must have one entry per class.")

    # Step 1: choose tau if needed
    if tau is None:
        if val_probs is None or y_val is None:
            raise ValueError("val_probs and y_val are required when tau is None.")

        tau, retain_fraction = choose_tau_from_validation(
            target_probs=target_probs,
            val_probs=val_probs,
            y_val=y_val,
            mode=threshold_mode,
        )
    else:
        retain_fraction = None

    target_conf = target_probs.max(axis=1)
    target_top = target_probs.argmax(axis=1)

    # Step 2: high-confidence set A
    A_mask = target_conf >= tau
    A_labels = target_top[A_mask]

    counts = np.full(n_classes, count_smoothing, dtype=float)

    if len(A_labels) > 0:
        counts += np.bincount(A_labels, minlength=n_classes)
        current_prior = counts / counts.sum()
    else:
        # Fallback if tau is too strict.
        # One could also use classify-and-count over the full target set.
        current_prior = source_prior.copy()

    # Step 3: low-confidence set B, sorted by decreasing confidence
    B_indices = np.where(~A_mask)[0]
    B_indices = B_indices[np.argsort(-target_conf[B_indices])]

    # Step 4: incremental pass over B
    incremental_labels = []

    for idx in B_indices:
        corrected = bayes_prior_update(
            probs=target_probs[idx],
            new_prior=current_prior,
            source_prior=source_prior,
            eps=eps,
        )
        pseudo_label = int(np.argmax(corrected))
        incremental_labels.append(pseudo_label)

        counts[pseudo_label] += 1.0
        current_prior = counts / counts.sum()

    estimated_intermediate_prior = current_prior.copy()

    # Step 5: final complete pass over all target instances
    corrected_all = bayes_prior_update(
        probs=target_probs,
        new_prior=estimated_intermediate_prior,
        source_prior=source_prior,
        eps=eps,
    )

    final_labels = corrected_all.argmax(axis=1)
    estimated_prior = np.bincount(final_labels, minlength=n_classes).astype(float)
    estimated_prior /= estimated_prior.sum()

    if not return_details:
        return estimated_prior

    details = {
        "tau": tau,
        "retain_fraction": retain_fraction,
        "n_A": int(A_mask.sum()),
        "n_B": int((~A_mask).sum()),
        "A_mask": A_mask,
        "intermediate_prior": estimated_intermediate_prior,
        "final_labels": final_labels,
        "corrected_probs": corrected_all,
        "incremental_labels": np.asarray(incremental_labels, dtype=int),
    }

    return estimated_prior, details