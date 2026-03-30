import numpy as np
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
    concordance_index_ipcw,
)


def discrete_rc_nll(events, labels, surv_bins, pad_left=1, pad_right=1, eps=1e-12):
    """
    Right-censored NLL for padded discrete survival outputs.

    Conventions
    -----------
    - labels are padded labels, e.g. first real bin has label = pad_left
    - surv_bins[:, 0] is the left dummy bin and should equal 1
    - real bins are [pad_left, ..., K_total - pad_right - 1]
    - right pad bins are ignored

    Returns
    -------
    nll_rc : float
        Mean right-censored negative log-likelihood.
    nll_per_real_bin : float
        NLL normalized by the number of real bins.
    """
    events = np.asarray(events, dtype=bool).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    S = np.asarray(surv_bins, dtype=float)

    n, K_total = S.shape
    if labels.shape[0] != n:
        raise ValueError("labels and surv_bins must have matching first dimension.")

    K_real = K_total - pad_left - pad_right
    if K_real <= 0:
        raise ValueError("No real bins remain after removing padding.")

    real_min = pad_left
    real_max = K_total - pad_right - 1
    if labels.min() < real_min or labels.max() > real_max:
        raise ValueError(f"labels must lie in [{real_min}, {real_max}]")

    # Enforce dummy left bin = 1 for likelihood semantics
    S = S.copy()
    S[:, 0] = 1.0

    # Optional monotonicity check on real bins
    S_real = S[:, pad_left:K_total - pad_right]
    if np.any(np.diff(S_real, axis=1) > 1e-8):
        raise ValueError("surv_bins must be non-increasing over real bins.")

    # Previous survival:
    # first real bin -> 1
    # later bins     -> previous padded column
    S_prev = np.ones(n, dtype=float)
    has_prev = labels > pad_left
    S_prev[has_prev] = S[np.arange(n)[has_prev], labels[has_prev] - 1]

    # Current survival at observed bin
    S_curr = S[np.arange(n), labels]

    # Event: P(T in bin y) = S_{y-1} - S_y
    prob_event = S_prev - S_curr

    # Censor: P(T > end of bin y) = S_y
    prob_cens = S_curr

    if np.any(prob_event < -1e-8):
        raise ValueError("Negative event probabilities found. Check survival indexing.")

    prob_event = np.clip(prob_event, eps, 1.0)
    prob_cens = np.clip(prob_cens, eps, 1.0)

    prob = np.where(events, prob_event, prob_cens)

    nll_rc = float(-np.log(prob).mean())
    nll_per_real_bin = nll_rc / float(K_real)

    return nll_rc, nll_per_real_bin


def _extract_subject_surv_at_own_time(test_times, surv_prob, eval_times):
    """
    Extract S_i(t_i) from survival matrix evaluated on eval_times.

    Parameters
    ----------
    test_times : (n,)
        Subject-specific observed times (event or censor time).
    surv_prob : (n, m)
        Survival probabilities evaluated at eval_times.
    eval_times : (m,)
        Time grid corresponding to surv_prob columns.

    Returns
    -------
    pred_probs : (n,)
        Subject-specific S_i(t_i), using right-continuous step evaluation.
    """
    test_times = np.asarray(test_times, dtype=float).reshape(-1)
    surv_prob = np.asarray(surv_prob, dtype=float)
    eval_times = np.asarray(eval_times, dtype=float).reshape(-1)

    n, m = surv_prob.shape
    if test_times.shape[0] != n:
        raise ValueError("test_times and surv_prob must have matching first dimension.")
    if eval_times.shape[0] != m:
        raise ValueError("eval_times and surv_prob must have matching second dimension.")

    # largest eval_time <= test_time
    idx = np.searchsorted(eval_times, test_times, side="right") - 1

    pred_probs = np.ones(n, dtype=float)  # before first grid point => survival = 1
    valid = idx >= 0
    pred_probs[valid] = surv_prob[np.arange(n)[valid], idx[valid]]

    return np.clip(pred_probs, 0.0, 1.0)


def _censored_dcal_contribution(s_prob: float, num_bins: int) -> np.ndarray:
    """
    Fractional D-cal histogram contribution for one censored subject.

    Under calibration:
        U = S(T) | T > C  ~ Uniform(0, S(C))
    """
    if not (0.0 <= s_prob <= 1.0):
        raise ValueError(f"s_prob must be in [0, 1], got {s_prob}")

    hist = np.zeros(num_bins, dtype=float)

    if np.isclose(s_prob, 0.0):
        hist[0] = 1.0
        return hist

    if np.isclose(s_prob, 1.0):
        hist[:] = 1.0 / num_bins
        return hist

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    for b in range(num_bins):
        left, right = edges[b], edges[b + 1]
        overlap = max(0.0, min(right, s_prob) - left)
        hist[b] = overlap / s_prob

    return hist


def d_calibration_error(pred_probs, event_indicators, num_bins=10, percentiles=None):
    """
    MSE-style D-calibration error:
        mean_rho (F_hat(rho) - rho)^2

    Parameters
    ----------
    pred_probs : (n,)
        Predicted survival probabilities at each subject's observed bin/time.
    event_indicators : (n,)
        1 for event, 0 for censored.
    """
    pred_probs = np.asarray(pred_probs, dtype=float).reshape(-1)
    event_indicators = np.asarray(event_indicators, dtype=int).reshape(-1)

    if pred_probs.shape[0] != event_indicators.shape[0]:
        raise ValueError("pred_probs and event_indicators must have same length.")

    pred_probs = np.clip(pred_probs, 0.0, 1.0)

    if percentiles is None:
        percentiles = np.arange(0.1, 1.0, 0.1)
    rho = np.asarray(percentiles, dtype=float)

    hist = np.zeros(num_bins, dtype=float)
    edges = np.linspace(0.0, 1.0, num_bins + 1)

    # Observed events: hard assignment
    obs_mask = event_indicators == 1
    obs_probs = pred_probs[obs_mask]
    # side='left' ensures p=0.1 counts in the first CDF bin at rho=0.1
    obs_bins = np.searchsorted(edges[1:], obs_probs, side="left")
    obs_bins = np.clip(obs_bins, 0, num_bins - 1)
    for b in obs_bins:
        hist[b] += 1.0

    # Censored subjects: fractional assignment
    cens_mask = event_indicators == 0
    for p in pred_probs[cens_mask]:
        hist += _censored_dcal_contribution(float(p), num_bins)

    # Normalize to histogram / empirical CDF
    hist = hist / pred_probs.shape[0]
    cdf = np.cumsum(hist)

    # Evaluate PP curve at requested rho
    bin_idx = np.searchsorted(edges[1:], rho, side="left")
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)
    pp_hat = cdf[bin_idx]

    dcal = float(np.mean((pp_hat - rho) ** 2))
    return dcal, pp_hat, rho, hist


def compute_surv_metrics(
    train_surv,
    test_surv,
    risk_prob,
    surv_prob,
    times,
    dcal_num_bins=10,
    dcal_percentiles=None,
):
    eps = 1e-12

    # ---- Plain C-index (full data) ----
    cindex, *_ = concordance_index_censored(
        test_surv["event"], test_surv["time"], risk_prob
    )

    # ---- D-Calibration on full test set ----
    pred_probs_dcal = _extract_subject_surv_at_own_time(
        test_times=test_surv["time"],
        surv_prob=surv_prob,
        eval_times=times,
    )
    dcal, dcal_pp, dcal_rho, dcal_hist = d_calibration_error(
        pred_probs=pred_probs_dcal,
        event_indicators=test_surv["event"].astype(int),
        num_bins=dcal_num_bins,
        percentiles=dcal_percentiles,
    )

    # ---- IPCW-safe test set ----
    test_ipcw, risk_ipcw, surv_ipcw = test_surv, risk_prob, surv_prob
    censor_mask = ~test_ipcw["event"]
    times_ipcw = np.asarray(times, dtype=float)

    if censor_mask.any():
        max_censor_time = test_ipcw["time"][censor_mask].max()
        drop_mask = (test_ipcw["event"]) & (test_ipcw["time"] >= max_censor_time)
        if drop_mask.any():
            print(
                f"\n[Warning] Dropping {drop_mask.sum()} event(s) at/after censoring "
                f"max={max_censor_time} for IPCW-based metrics."
            )
            test_ipcw = test_ipcw[~drop_mask]
            risk_ipcw = risk_ipcw[~drop_mask]
            surv_ipcw = surv_ipcw[~drop_mask]

        max_time_ipcw = test_ipcw["time"].max()
        keep = times_ipcw < max_time_ipcw
        times_ipcw = times_ipcw[keep]
        surv_ipcw = surv_ipcw[:, keep]

    # ---- IPCW C-index ----
    cindex_ipcw, *_ = concordance_index_ipcw(train_surv, test_ipcw, risk_ipcw)

    # ---- Integrated Brier Score ----
    ibs = integrated_brier_score(train_surv, test_ipcw, surv_ipcw, times_ipcw)

    # ---- Time-dependent AUC ----
    auc, mean_auc = cumulative_dynamic_auc(train_surv, test_ipcw, 1 - surv_ipcw, times_ipcw)
    if np.isnan(mean_auc):
        raise ValueError("AUC is NaN — check survival probabilities or time points.")

    # ---- Collect metrics ----
    metrics = {
        "C-index": cindex,
        "C-index IPCW": cindex_ipcw,
        "AUC": mean_auc,
        "IBS": ibs,
        "DCal": dcal,            
    }
    return metrics