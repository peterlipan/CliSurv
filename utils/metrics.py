import numpy as np
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score, concordance_index_ipcw


def discrete_rc_nll(events, labels, surv_bins, eps=1e-12):
    """
    Right-censored NLL on dataset's discrete time bins.

    Parameters
    ----------
    events    : array-like, shape (n,)
        Boolean or {0,1}. True/1 = event, False/0 = censored.
    labels    : array-like, shape (n,)
        Integer bin indices in [0, K-1], as used in training (data['label']).
    surv_bins : array-like, shape (n, K)
        Predicted survival probabilities at the end of each bin.
        This is outputs.surv evaluated on the dataset's K bins.

    eps : float
        Small constant to avoid log(0) or log of negative due to numerical issues.

    Returns
    -------
    NLL_rc      : float
        Mean negative log-likelihood over all test samples.
    NLL_per_bin : float
        NLL_rc divided by K (optional normalization for comparability if K differs).
    """
    events = np.asarray(events, dtype=bool)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    S = np.asarray(surv_bins, dtype=float)

    n, K = S.shape
    assert labels.shape[0] == n
    assert labels.min() >= 0 and labels.max() < K

    # S_prev_i = S_i(label_i - 1) for label>0, else 1 for label=0
    S_prev = np.ones(n, dtype=float)
    mask_gt0 = labels > 0
    S_prev[mask_gt0] = S[mask_gt0, labels[mask_gt0] - 1]

    # S_curr_i = S_i(label_i)
    S_curr = S[np.arange(n), labels]

    # Event probability: P(T in bin y) = S_prev - S_curr
    prob_event = S_prev - S_curr

    # Censor probability: P(T > end of bin y) = S_curr
    prob_cens = S_curr.copy()

    # Numerical safety only: clip to [eps, 1] so log() is defined
    prob_event = np.clip(prob_event, eps, 1.0)
    prob_cens  = np.clip(prob_cens,  eps, 1.0)

    # Build per-sample likelihood
    events_f = events.astype(float)
    prob = events_f * prob_event + (1.0 - events_f) * prob_cens

    # Final NLL
    logL = np.log(prob)
    NLL_rc = float(-logL.mean())
    NLL_per_bin = NLL_rc / float(K)

    return NLL_rc, NLL_per_bin



def compute_surv_metrics(train_surv, test_surv, risk_prob, surv_prob, times):
    eps = 1e-12

    # ---- Plain C-index (full data) ----
    cindex, *_ = concordance_index_censored(
        test_surv["event"], test_surv["time"], risk_prob
    )

    # ---- IPCW-safe test set ----
    test_ipcw, risk_ipcw, surv_ipcw = test_surv, risk_prob, surv_prob
    censor_mask = ~test_ipcw["event"]

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

        # ensure times fit within new support
        max_time_ipcw = test_ipcw["time"].max()
        times = times[times < max_time_ipcw]

    # ---- IPCW C-index ----
    cindex_ipcw, *_ = concordance_index_ipcw(train_surv, test_ipcw, risk_ipcw)

    # ---- Integrated Brier Score ----
    ibs = integrated_brier_score(train_surv, test_ipcw, surv_ipcw, times)

    # ---- Time-dependent AUC ----
    auc, mean_auc = cumulative_dynamic_auc(train_surv, test_ipcw, 1 - surv_ipcw, times)
    if np.isnan(mean_auc):
        raise ValueError("AUC is NaN — check survival probabilities or time points.")


    # ---- Collect metrics ----
    metrics = {
        "C-index": cindex,
        "C-index IPCW": cindex_ipcw,
        "AUC": mean_auc,
        "IBS": ibs,
    }
    return metrics
