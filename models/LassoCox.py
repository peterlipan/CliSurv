import torch
import numpy as np
import torch.nn as nn
from .backbone import get_encoder
from .utils import ModelOutputs
from pycox.models.loss import CoxPHLoss


class CoxSurvLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.cph = CoxPHLoss()
        self.eps = eps
    def forward(self, outputs, data):
        return self.cph(outputs.risk, data['duration'], data['event'])


class LassoCox(nn.Module):
    """
    L1-regularized Cox proportional hazards (Lasso-Cox) with the same public
    interface and outputs as your DeepSurv model.

    Notes:
      - For a true "lasso on covariates", use an identity encoder so that the
        Linear head sees raw features.
      - L1 is applied only to the head weights (not the bias or encoder).
    """
    def __init__(self, args):
        super().__init__()
        self.head = nn.Linear(args.n_features, 1, bias=True)

        # L1 strength (lambda). Provide as args.l1 or default small.
        self.l1 = getattr(args, 'l1', 1e-4)

        # Cox loss (no L1 inside)
        self.criterion = CoxSurvLoss()

        # Baseline storage (for survival curve estimation) — mirrors DeepSurv
        self.event_times_ = None
        self.baseline_cum_hazard_ = None
        self.baseline_surv_ = None
        self.lp_mean_ = None
        self.centered_ = False

        # Discrete time grid (bins) for evaluation
        self.bin_times_ = None
        self.baseline_surv_bins_ = None
        self._baseline_ready_for_bins = False

        self.pad_left = args.pad_left

    def forward(self, data):
        x = data['data']
        logits = self.head(x)
        risk = logits.view(-1)

        surv = None
        if self.baseline_surv_bins_ is not None:
            lp_centered = risk - self.lp_mean_ if self.centered_ and self.lp_mean_ is not None else risk
            hr = torch.exp(lp_centered).unsqueeze(1)  # (batch, 1)
            surv_baseline = self.baseline_surv_bins_.to(risk.device).unsqueeze(0)  # (1, n_bins)
            surv = surv_baseline ** hr

        return ModelOutputs(features=x, logits=logits, risk=risk, surv=surv)

    def compute_loss(self, outputs, data):
        # Cox partial likelihood loss
        cox_loss = self.criterion(outputs, data)
        # L1 penalty on the head weights only (exclude bias)
        l1_pen = self.head.weight.abs().sum()
        return cox_loss + self.l1 * l1_pen

    # ---------- Baseline estimation & utilities (exactly mirrored behavior) ----------

    @torch.no_grad()
    def configure_time_bins(self, bin_times):
        if not torch.is_tensor(bin_times):
            bin_times = torch.tensor(bin_times, dtype=torch.float32)
        self.bin_times_ = bin_times.float().clone()
        if self.baseline_surv_ is not None:
            self._project_baseline_to_bins()
        return self

    @torch.no_grad()
    def estimate_baseline_surv(self, train_loader, device=None, center=True, eps=1e-12):
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

        if self.bin_times_ is not None:
            self._project_baseline_to_bins()

        if was_training:
            self.train()

        return event_times, H0, S0

    @torch.no_grad()
    def _project_baseline_to_bins(self):
        if self.baseline_surv_ is None or self.event_times_ is None:
            raise RuntimeError("Baseline survival not estimated yet.")
        if self.bin_times_ is None:
            raise RuntimeError("Bin times not configured.")

        et = self.event_times_.cpu()
        S0 = self.baseline_surv_.cpu()
        bt = self.bin_times_.cpu()

        idx = torch.searchsorted(et, bt, right=True) - 1
        baseline_bins = torch.ones_like(bt)
        valid = idx >= 0
        baseline_bins[valid] = S0[idx[valid]]
        baseline_bins = baseline_bins.clamp(min=1e-12, max=1.0)

        self.baseline_surv_bins_ = baseline_bins
        self._baseline_ready_for_bins = True

    @torch.no_grad()
    def prepare_for_validation(self, train_loader, bin_times, device=None, force=False):
        if force or self.baseline_surv_ is None:
            self.estimate_baseline_surv(train_loader, device=device)
        self.configure_time_bins(bin_times)

    @torch.no_grad()
    def predict_survival_matrix(self, data_loader, device=None):
        if self.baseline_surv_bins_ is None:
            raise RuntimeError("Baseline not projected to bins. Call prepare_for_validation.")
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        surv_rows = []
        for batch in data_loader:
            x = batch['data'].to(device)
            out = self.forward({'data': x})
            if out.surv is None:
                raise RuntimeError("Forward did not produce surv. Baseline not ready?")
            surv_rows.append(out.surv.cpu())
        return torch.cat(surv_rows, dim=0)

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
            "method": "LassoCox",
            "baseline_survival": S0,
            "ground_truth_survival": ground_truth_survival,
        }
        return payload


