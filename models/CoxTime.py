import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
from .utils import ModelOutputs


class CoxTimeLoss(nn.Module):
    """
    Discrete-time Cox (time-dependent) partial likelihood.

    For each bin k with at least one event:
        sum_{i: event_i=1, label_i=k} f(x_i, t_k)
        - (#events at k) * logsumexp_{j in R_k} f(x_j, t_k),
    where the risk set R_k = { j : label_j >= k }.
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, K)   time-dependent log-risk f(x, t_k)
        labels: (B,)     integer time bin in [0..K-1]
        events: (B,)     0/1
        """
        B, K = logits.shape
        labels = labels.view(-1).long()
        events = events.view(-1).float()

        loss = logits.new_tensor(0.)
        n_events_total = (events == 1).sum().clamp(min=1)

        for k in range(K):
            ev_mask = (labels == k) & (events == 1)
            n_ev_k = ev_mask.sum()
            if n_ev_k == 0:
                continue

            numer = logits[ev_mask, k].sum()
            risk_mask = (labels >= k)
            denom_log = torch.logsumexp(logits[risk_mask, k], dim=0)
            loss = loss - (numer - n_ev_k * denom_log)

        return loss / n_events_total


class CoxTime(nn.Module):
    """
    CoxTime with discrete time bins (K = args.n_classes).

    Inputs (batch):
        data['data']  : features for encoder
        data['label'] : int bin in [0..K-1]
        data['event'] : 0/1

    Outputs (ModelOutputs):
        features: [B, d_hid]
        logits  : [B, K] (f(x, t_k))
        risk    : [B]    (negative expected time if baseline ready, else proxy)
        surv    : [B, K] (only after baseline is prepared)
    """
    def __init__(self, args):
        super().__init__()
        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = int(args.n_classes)

        # Time-dependent head: f(x, t_k) for all k in one pass
        self.head = nn.Linear(self.d_hid, self.n_classes, bias=True)
        self.criterion = CoxTimeLoss()

        # ---- Baseline storage (harmonized with your DeepSurv API names) ----
        self.event_times_ = None               # here: integer bin indices 0..K-1
        self.baseline_cum_hazard_ = None       # H0[k] = sum_{m<=k} ΔH0[m]
        self.baseline_surv_ = None             # S0[k] = exp(-H0[k]) at bin indices
        self.lp_mean_ = None                   # unused for CoxTime (kept for parity)
        self.centered_ = False

        # Discrete time grid for evaluation (user-facing bin coordinates)
        self.bin_times_ = None                 # tensor (K,)
        self.baseline_surv_bins_ = None        # same as baseline_surv_ but aligned to bin_times_
        self._baseline_ready_for_bins = False

        # Internal: baseline increments ΔH0[k]
        self._delta_H0_ = None                 # (K,)

    # ---- Forward & training loss ----
    def forward(self, data):
        features = self.encoder(data['data'])
        logits = self.head(features)  # (B, K)

        surv = None
        if self._baseline_ready_for_bins and self._delta_H0_ is not None:
            # Per-sample cumulative hazard with time-dependent effects:
            # ΔH_i[k] = ΔH0[k] * exp(f(x_i, t_k))
            exp_logits = torch.exp(logits)                               # (B, K)
            delta_H_i = self._delta_H0_.to(logits.device).unsqueeze(0) * exp_logits
            H_i = torch.cumsum(delta_H_i, dim=1)                         # (B, K)
            surv = torch.exp(-H_i)                                       # (B, K)

        # Risk score for C-index:
        # If surv is available, use negative expected time from PMF; else use a proxy.
        if surv is not None:
            pmf = torch.zeros_like(surv)
            pmf[:, 0] = 1.0 - surv[:, 0]
            if self.n_classes > 1:
                pmf[:, 1:] = surv[:, :-1] - surv[:, 1:]
            time_bins = torch.arange(1, self.n_classes + 1, device=logits.device, dtype=logits.dtype)
            risk = -(pmf * time_bins.unsqueeze(0)).sum(dim=1)
        else:
            # Proxy: average log-risk over bins
            risk = logits.mean(dim=1)

        return ModelOutputs(features=features, logits=logits, risk=risk, surv=surv)

    def compute_loss(self, outputs, data):
        return self.criterion(outputs.logits, data['label'], data['event'])

    # ---- Baseline estimation & utilities (names mirror DeepSurv) ----
    @torch.no_grad()
    def configure_time_bins(self, bin_times):
        """
        Set the discrete evaluation time grid (length K).
        These are coordinates for reporting/plotting; computations use bin indices.
        """
        if not torch.is_tensor(bin_times):
            bin_times = torch.tensor(bin_times, dtype=torch.float32)
        if bin_times.numel() != self.n_classes:
            raise ValueError(f"bin_times must have length {self.n_classes}")
        self.bin_times_ = bin_times.float().clone()
        # If we already have a baseline S0 at bin indices, project it:
        if self.baseline_surv_ is not None:
            self._project_baseline_to_bins()
        return self

    @torch.no_grad()
    def estimate_baseline_surv(self, train_loader, device=None, eps: float = 1e-12):
        """
        Breslow baseline for CoxTime (discrete time):
            ΔH0[k] = d_k / sum_{j in R_k} exp(f(x_j, t_k))
        with d_k = #events at bin k, R_k = { j : label_j >= k }.

        H0[k] = cumulative sum of ΔH0 up to k,
        S0[k] = exp(-H0[k]).
        """
        was_training = self.training
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        logits_list, labels_list, events_list = [], [], []
        for batch in train_loader:
            x = batch['data'].to(device)
            feats = self.encoder(x)
            logits_list.append(self.head(feats).detach().cpu())   # (B, K)
            labels_list.append(batch['label'].detach().cpu().long().view(-1))
            events_list.append(batch['event'].detach().cpu().float().view(-1))

        logits = torch.cat(logits_list, dim=0)   # (N, K)
        labels = torch.cat(labels_list, dim=0)   # (N,)
        events = torch.cat(events_list, dim=0)   # (N,)

        N, K = logits.shape
        exp_logits = torch.exp(logits)           # (N, K)

        # Events per bin
        d_k = torch.zeros(K)
        for k in range(K):
            d_k[k] = ((labels == k) & (events == 1)).sum()

        # Risk set sum per bin
        risk_sums = torch.zeros(K)
        for k in range(K):
            mask = (labels >= k)
            risk_sums[k] = exp_logits[mask, k].sum() if mask.any() else 0.0

        delta_H0 = d_k / (risk_sums + eps)          # (K,)
        H0 = torch.cumsum(delta_H0, dim=0)          # (K,)
        S0 = torch.exp(-H0)                         # (K,)

        # Store with the same field names as DeepSurv
        self.event_times_ = torch.arange(K).float() # bin indices as "event times"
        self._delta_H0_ = delta_H0
        self.baseline_cum_hazard_ = H0
        self.baseline_surv_ = S0
        self._baseline_ready_for_bins = False

        # If bin grid already configured, project now
        if self.bin_times_ is not None:
            self._project_baseline_to_bins()

        if was_training:
            self.train()

        return self.event_times_, H0, S0

    @torch.no_grad()
    def _project_baseline_to_bins(self):
        """
        For CoxTime, baseline S0 is *already* defined at integer bin indices 0..K-1.
        When users provide custom bin_times_, we simply align by index (no interpolation),
        and expose S0 at those reporting times.
        """
        if self.baseline_surv_ is None or self.event_times_ is None:
            raise RuntimeError("Baseline survival not estimated yet.")
        if self.bin_times_ is None:
            raise RuntimeError("Bin times not configured.")

        # Index-aligned projection (bins correspond 1-to-1)
        S0 = self.baseline_surv_.cpu()
        bt = self.bin_times_.cpu()
        if S0.numel() != bt.numel():
            raise RuntimeError("bin_times length must equal number of bins (n_classes).")

        baseline_bins = S0.clone()
        baseline_bins = baseline_bins.clamp(min=1e-12, max=1.0)

        self.baseline_surv_bins_ = baseline_bins
        self._baseline_ready_for_bins = True

    @torch.no_grad()
    def prepare_for_validation(self, train_loader, bin_times, device=None, force=False):
        """
        Convenience: estimate baseline (if not done or force=True) and set bin grid.
        """
        if force or self.baseline_surv_ is None:
            self.estimate_baseline_surv(train_loader, device=device)
        self.configure_time_bins(bin_times)

    @torch.no_grad()
    def predict_survival_matrix(self, data_loader, device=None):
        """
        Returns survival matrix (n_samples, K) using precomputed ΔH0.
        """
        if not self._baseline_ready_for_bins or self._delta_H0_ is None:
            raise RuntimeError("Baseline not projected to bins. Call prepare_for_validation.")
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        rows = []
        for batch in data_loader:
            x = batch['data'].to(device)
            feats = self.encoder(x)
            logits = self.head(feats)                  # (B, K)
            exp_logits = torch.exp(logits)
            delta_H_i = self._delta_H0_.to(logits.device).unsqueeze(0) * exp_logits
            H_i = torch.cumsum(delta_H_i, dim=1)
            S_i = torch.exp(-H_i)
            rows.append(S_i.cpu())
        return torch.cat(rows, dim=0)
