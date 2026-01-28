import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
from .utils import ModelOutputs
from pycox.models.loss import CoxPHLoss
import torchtuples as tt
from pycox.models import CoxPH


class CoxSurvLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.cph = CoxPHLoss()
        self.eps = eps
    def forward(self, outputs, data):
        return self.cph(outputs.risk, data['duration'], data['event'])


class DeepSurv(nn.Module):
    def __init__(self, args):
        super(DeepSurv, self).__init__()
        
        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes
        self.head = nn.Linear(self.d_hid, 1)
        self.criterion = CoxSurvLoss()
            # Baseline storage
        self.event_times_ = None                  # distinct event times (tensor)
        self.baseline_cum_hazard_ = None          # cumulative hazard at event times
        self.baseline_surv_ = None                # baseline survival at event times
        self.lp_mean_ = None                      # mean of lp if centered
        self.centered_ = False

        # Discrete time grid (bins) for evaluation
        self.bin_times_ = None                    # tensor (n_bins,)
        self.baseline_surv_bins_ = None           # baseline survival projected onto bin times
        self._baseline_ready_for_bins = False

    def forward(self, data):
        features = self.encoder(data['data'])
        logits = self.head(features)
        risk = logits.view(-1)  # linear predictor (log hazard ratio)

        surv = None
        if self.baseline_surv_bins_ is not None:
            # Use centered linear predictor for survival estimation
            lp_centered = risk - self.lp_mean_ if self.centered_ and self.lp_mean_ is not None else risk
            hr = torch.exp(lp_centered).unsqueeze(1)  # (batch, 1)
            # baseline_surv_bins_: (n_bins,)
            surv_baseline = self.baseline_surv_bins_.to(risk.device).unsqueeze(0)  # (1, n_bins)
            surv = surv_baseline ** hr   # S(t|x) = S0(t)^{exp(lp_centered)}

        return ModelOutputs(features=features, logits=logits, risk=risk, surv=surv)

    def compute_loss(self, outputs, data):
        return self.criterion(outputs, data)

    @torch.no_grad()
    def configure_time_bins(self, bin_times):
        """
        Set the discrete evaluation time grid (length n_bins).
        bin_times: 1D array-like sorted ascending (float or int).
        """
        if not torch.is_tensor(bin_times):
            bin_times = torch.tensor(bin_times, dtype=torch.float32)
        self.bin_times_ = bin_times.float().clone()
        # If we already have a baseline survival at event_times_, project it:
        if self.baseline_surv_ is not None:
            self._project_baseline_to_bins()
        return self

    @torch.no_grad()
    def estimate_baseline_surv(self, train_loader, device=None, center=True, eps=1e-12):
        """
        Breslow baseline cumulative hazard + survival. Must be called after training.
        """
        was_training = self.training
        self.eval()

        if device is None:
            device = next(self.parameters()).device

        durations_list, events_list, lp_list = [], [], []
        for batch in train_loader:
            x = batch['data'].to(device)
            out = self.forward({'data': x})
            lp_list.append(out.risk.detach().cpu())
            durations_list.append(batch['duration'].detach().cpu())
            events_list.append(batch['event'].detach().cpu())

        durations = torch.cat(durations_list).float()
        events = torch.cat(events_list).int()
        lp = torch.cat(lp_list).float()

        if (events == 1).sum() == 0:
            raise ValueError("No events in training data; cannot estimate baseline hazard.")

        self.lp_mean_ = lp.mean() if center else torch.tensor(0.0)
        if center:
            lp = lp - self.lp_mean_
        self.centered_ = center

        order = torch.argsort(durations)
        t_sorted = durations[order]
        e_sorted = events[order]
        lp_sorted = lp[order]

        event_mask = (e_sorted == 1)
        event_times = torch.unique(t_sorted[event_mask])

        exp_lp_sorted = torch.exp(lp_sorted)
        rev_cumsum = torch.cumsum(exp_lp_sorted.flip(0), dim=0).flip(0)

        idx_first = torch.searchsorted(t_sorted, event_times, right=False)
        # Map each event row to its event_time index for d_j
        mapping = torch.searchsorted(event_times, t_sorted[event_mask])
        d_counts = torch.bincount(mapping, minlength=event_times.shape[0]).float()

        risk_sums = rev_cumsum[idx_first]
        increments = d_counts / (risk_sums + eps)
        H0 = torch.cumsum(increments, dim=0)
        S0 = torch.exp(-H0)

        self.event_times_ = event_times
        self.baseline_cum_hazard_ = H0
        self.baseline_surv_ = S0
        self._baseline_ready_for_bins = False

        # If bin grid already configured, project now
        if self.bin_times_ is not None:
            self._project_baseline_to_bins()

        if was_training:
            self.train()

        return event_times, H0, S0

    @torch.no_grad()
    def _project_baseline_to_bins(self):
        """
        Project the stepwise baseline survival defined at event_times_
        onto the predefined bin_times_ (right-continuous survival function).
        """
        if self.baseline_surv_ is None or self.event_times_ is None:
            raise RuntimeError("Baseline survival not estimated yet.")
        if self.bin_times_ is None:
            raise RuntimeError("Bin times not configured.")

        et = self.event_times_.cpu()
        S0 = self.baseline_surv_.cpu()
        bt = self.bin_times_.cpu()

        # For each bin time t, find largest event_time <= t.
        # searchsorted(et, t, right=True) gives insertion point; subtract 1.
        idx = torch.searchsorted(et, bt, right=True) - 1
        # Where idx < 0 (before first event), survival = 1.0
        baseline_bins = torch.ones_like(bt)
        valid = idx >= 0
        baseline_bins[valid] = S0[idx[valid]]

        # (Optional) small clamp to avoid exact 0 or 1 extremes
        baseline_bins = baseline_bins.clamp(min=1e-12, max=1.0)

        self.baseline_surv_bins_ = baseline_bins
        self._baseline_ready_for_bins = True

    @torch.no_grad()
    def prepare_for_validation(self, train_loader, bin_times, device=None, force=False):
        """
        Convenience: estimate baseline (if not done or force=True) and set bin grid.
        bin_times: array-like length n_bins.
        Call this once before running surv_validate().
        """
        if force or self.baseline_surv_ is None:
            self.estimate_baseline_surv(train_loader, device=device)
        self.configure_time_bins(bin_times)

    @torch.no_grad()
    def predict_survival_matrix(self, data_loader, device=None):
        """
        Returns survival matrix (n_samples, n_bins) using precomputed baseline.
        """
        if self.baseline_surv_bins_ is None:
            raise RuntimeError("Baseline not projected to bins. Call prepare_for_validation.")
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        surv_rows = []
        for batch in data_loader:
            x = batch['data'].to(device)
            out = self.forward({'data': x})  # forward supplies surv if baseline ready
            if out.surv is None:
                raise RuntimeError("Forward did not produce surv. Baseline not ready?")
            surv_rows.append(out.surv.cpu())
        return torch.cat(surv_rows, dim=0)  # (n, n_bins)

    @torch.no_grad()
    def save_baseline(
        self,
        ground_truth_survival: np.ndarray,
    ):
        # Must have projected baseline already
        if self.baseline_surv_bins_ is None:
            raise RuntimeError(
                "baseline_surv_bins_ is None. Call prepare_for_validation(...) first."
            )

        ground_truth_survival = np.asarray(ground_truth_survival, dtype=np.float32)
        if ground_truth_survival.ndim != 1:
            raise ValueError("ground_truth_survival must be 1D")

        S0 = self.baseline_surv_bins_.detach().cpu().numpy().astype(np.float32)  # [T]

        if S0.shape[0] != ground_truth_survival.shape[0]:
            raise ValueError(
                f"baseline_surv_bins_ length ({S0.shape[0]}) must match "
                f"ground_truth_survival ({ground_truth_survival.shape[0]})"
            )

        payload = {
            "method": "DeepSurv",
            "baseline_survival": S0,
            "ground_truth_survival": ground_truth_survival,
        }
        return payload