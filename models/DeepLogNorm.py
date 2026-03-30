import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .backbone import get_encoder
from .utils import ModelOutputs


# =========================
# LOGNORMAL LOSS
# =========================
class LogNormalSurvLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, outputs, data):
        y = data['duration'].view(-1).float()
        event = data['event'].view(-1).float()

        mu = outputs.mu.view(-1)
        sigma = outputs.sigma.view(-1)

        pred_dist = torch.distributions.LogNormal(mu, sigma)

        logpdf = pred_dist.log_prob(y)
        cdf = pred_dist.cdf(y)
        logsurv = torch.log(torch.clamp(1.0 - cdf, min=self.eps))

        loglike = event * logpdf + (1.0 - event) * logsurv
        return -loglike.mean()


# =========================
# LOGNORMAL MODEL
# =========================
class DeepLogNorm(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid

        self.mu_head = nn.Linear(self.d_hid, 1)
        self.sigma_head = nn.Linear(self.d_hid, 1)

        self.criterion = LogNormalSurvLoss()

        # Discrete time grid for evaluation (mirrors DeepSurv/DeepALD interface)
        self.bin_times_ = None          # tensor (n_bins,)

    def forward(self, data):
        features = self.encoder(data['data'])

        mu = self.mu_head(features)
        sigma = F.softplus(self.sigma_head(features)) + 1e-8

        logits = torch.cat([mu, sigma], dim=1)

        # earlier predicted time => higher risk
        risk = -mu.view(-1)

        # Survival matrix (B, n_bins) — available once bin_times_ is configured
        surv = None
        if self.bin_times_ is not None:
            surv = self._compute_survival_matrix(mu, sigma)

        return ModelOutputs(
            features=features,
            logits=logits,
            mu=mu,
            sigma=sigma,
            risk=risk,
            surv=surv,
        )

    def compute_loss(self, outputs, data):
        return self.criterion(outputs, data)

    # ------------------------------------------------------------------
    # Validation interface (mirrors DeepSurv / DeepALD)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def configure_time_bins(self, bin_times):
        """
        Set the discrete evaluation time grid (length n_bins).
        bin_times: 1D array-like sorted ascending (float or int).
        """
        if not torch.is_tensor(bin_times):
            bin_times = torch.tensor(bin_times, dtype=torch.float32)
        self.bin_times_ = bin_times.float().clone()
        return self

    @torch.no_grad()
    def prepare_for_validation(self, train_loader, bin_times, device=None, force=False):
        """
        Mirror of DeepSurv.prepare_for_validation().
        For DeepLogNorm the survival function is analytic, so no baseline
        estimation is needed — we only store the bin time grid.
        train_loader and force are accepted for API compatibility but unused.
        """
        self.configure_time_bins(bin_times)

    @torch.no_grad()
    def predict_survival_matrix(self, data_loader, device=None):
        """
        Returns survival matrix (n_samples, n_bins) using the analytic log-normal survival.
        Mirrors DeepSurv.predict_survival_matrix().
        """
        if self.bin_times_ is None:
            raise RuntimeError("Bin times not configured. Call prepare_for_validation first.")
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        surv_rows = []
        for batch in data_loader:
            x = batch['data'].to(device)
            out = self.forward({'data': x})
            if out.surv is None:
                raise RuntimeError("Forward did not produce surv. Are bin_times_ set?")
            surv_rows.append(out.surv.cpu())
        return torch.cat(surv_rows, dim=0)  # (n, n_bins)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_survival_matrix(self, mu, sigma):
        """
        Analytic log-normal survival S(t|x) = 1 - F(t|x) evaluated at self.bin_times_.
        mu, sigma: (B, 1)
        returns: (B, n_bins)
        """
        times = self.bin_times_.to(mu.device)   # (T,)
        B = mu.shape[0]
        T = times.shape[0]

        # Broadcast to (B, T)
        t   = times.unsqueeze(0).expand(B, T)
        mu_ = mu.expand(B, T)
        sig = sigma.expand(B, T)

        dist = torch.distributions.LogNormal(mu_, sig)
        cdf  = dist.cdf(t)
        surv = (1.0 - cdf).clamp(min=1e-8, max=1.0)
        return surv   # (B, T)

    @torch.no_grad()
    def predict_cdf(self, outputs, times):
        mu = outputs.mu.view(-1, 1)
        sigma = outputs.sigma.view(-1, 1)

        if times.ndim == 1:
            times = times.unsqueeze(0).repeat(mu.shape[0], 1)

        dist = torch.distributions.LogNormal(mu, sigma)
        return dist.cdf(times).clamp(min=1e-8, max=1.0)

    @torch.no_grad()
    def predict_survival(self, outputs, times):
        cdf = self.predict_cdf(outputs, times)
        return (1.0 - cdf).clamp(min=1e-8, max=1.0)