import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .backbone import get_encoder
from .utils import ModelOutputs


def safe_log(x, min_value=1e-8):
    return torch.log(torch.clamp(x, min=min_value))


# =========================
# ALD LOSS
# =========================
class ALDSurvLoss(nn.Module):
    def __init__(self, use_censor_loss=True, eps=1e-8):
        super().__init__()
        self.use_censor_loss = use_censor_loss
        self.eps = eps

    def forward(self, outputs, data):
        y = data['duration'].view(-1, 1).float()
        cen_indicator = (1 - data['event']).view(-1, 1).float()  # event=1 observed -> cen=0

        theta = outputs.theta
        sigma = outputs.sigma
        kappa = outputs.kappa

        sqrt2 = torch.sqrt(torch.tensor(2.0, device=y.device, dtype=y.dtype))

        alpha = torch.where(y >= theta, y - theta, torch.zeros_like(y))
        beta  = torch.where(y < theta, theta - y, torch.zeros_like(y))

        # observed event: negative log density
        obs_term = (
            safe_log(sigma, self.eps)
            - safe_log(kappa / (1.0 + kappa**2), self.eps)
            + (sqrt2 / sigma) * (alpha * kappa + beta / kappa)
        )

        loss_obs = (1.0 - cen_indicator) * obs_term

        if not self.use_censor_loss:
            return loss_obs.mean()

        # censored: negative log survival = -log(1 - F(y|x))
        loss_cen_1 = safe_log(1.0 + kappa**2, self.eps) + sqrt2 * kappa * (y - theta) / sigma

        value_to_exp = -sqrt2 * (theta - y) / (sigma * kappa)
        safe_exp_value = torch.clamp(value_to_exp, max=80.0)
        inner = 1.0 - (kappa**2 * torch.exp(safe_exp_value) / (1.0 + kappa**2))
        loss_cen_2 = -safe_log(inner, self.eps)

        cen_term = torch.where(y > theta, loss_cen_1, loss_cen_2)
        loss_cen = cen_indicator * cen_term

        return (loss_obs + loss_cen).mean()


# =========================
# ALD MODEL
# =========================
class DeepALD(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.use_dropout = getattr(args, 'dropout', 0.0) > 0
        self.dropout = nn.Dropout(getattr(args, 'dropout', 0.0))

        # parameter heads
        self.theta_head = nn.Linear(self.d_hid, 1)
        self.sigma_head = nn.Linear(self.d_hid, 1)
        self.kappa_head = nn.Linear(self.d_hid, 1)

        self.criterion = ALDSurvLoss(
            use_censor_loss=getattr(args, 'use_censor_loss', True)
        )

        # Discrete time grid for evaluation (mirrors DeepSurv interface)
        self.bin_times_ = None          # tensor (n_bins,)

    def forward(self, data):
        features = self.encoder(data['data'])
        if self.use_dropout:
            features = self.dropout(features)

        theta = self.theta_head(features)                          # unconstrained  (B, 1)
        sigma = F.softplus(self.sigma_head(features)) + 1e-8      # positive        (B, 1)
        kappa = F.softplus(self.kappa_head(features)) + 1e-8      # positive        (B, 1)

        logits = torch.cat([theta, sigma, kappa], dim=1)

        # risk for ranking: earlier predicted time => higher risk
        risk = -theta.view(-1)

        # Survival matrix (B, n_bins) — available once bin_times_ is configured
        surv = None
        if self.bin_times_ is not None:
            surv = self._compute_survival_matrix(theta, sigma, kappa)

        return ModelOutputs(
            features=features,
            logits=logits,
            theta=theta,
            sigma=sigma,
            kappa=kappa,
            risk=risk,
            surv=surv,
        )

    def compute_loss(self, outputs, data):
        return self.criterion(outputs, data)

    # ------------------------------------------------------------------
    # Validation interface (mirrors DeepSurv)
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
        For DeepALD the survival function is analytic, so no baseline
        estimation is needed — we only need to store the bin time grid.
        train_loader and force are accepted for API compatibility but unused.
        """
        self.configure_time_bins(bin_times)

    @torch.no_grad()
    def predict_survival_matrix(self, data_loader, device=None):
        """
        Returns survival matrix (n_samples, n_bins) using the analytic ALD survival.
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

    def _compute_survival_matrix(self, theta, sigma, kappa):
        """
        Analytic ALD survival S(t|x) = 1 - F(t|x) evaluated at self.bin_times_.
        theta, sigma, kappa: (B, 1)
        returns: (B, n_bins)
        """
        times = self.bin_times_.to(theta.device)   # (T,)
        T = times.shape[0]
        B = theta.shape[0]

        # Broadcast to (B, T)
        t   = times.unsqueeze(0).expand(B, T)
        th  = theta.expand(B, T)
        sig = sigma.expand(B, T)
        kap = kappa.expand(B, T)

        sqrt2 = torch.sqrt(torch.tensor(2.0, device=theta.device, dtype=theta.dtype))

        cdf_upper = 1.0 - 1.0 / (1.0 + kap**2) * torch.exp(-sqrt2 * kap * (t - th) / sig)
        cdf_lower = (kap**2 / (1.0 + kap**2)) * torch.exp(-sqrt2 * (th - t) / (sig * kap))

        cdf  = torch.where(t > th, cdf_upper, cdf_lower)
        surv = (1.0 - cdf).clamp(min=1e-8, max=1.0)
        return surv   # (B, T)

    @torch.no_grad()
    def predict_cdf(self, outputs, times):
        """
        outputs: ModelOutputs from forward()
        times: tensor of shape [T] or [B, T]
        returns cdf [B, T]
        """
        theta, sigma, kappa = outputs.theta, outputs.sigma, outputs.kappa
        B = theta.shape[0]

        if times.ndim == 1:
            times = times.unsqueeze(0).repeat(B, 1)

        theta = theta.expand_as(times)
        sigma = sigma.expand_as(times)
        kappa = kappa.expand_as(times)

        sqrt2 = torch.sqrt(torch.tensor(2.0, device=times.device, dtype=times.dtype))

        cdf_1 = 1.0 - 1.0 / (1.0 + kappa**2) * torch.exp(-sqrt2 * kappa * (times - theta) / sigma)
        cdf_2 = (kappa**2 / (1.0 + kappa**2)) * torch.exp(-sqrt2 * (theta - times) / (sigma * kappa))

        cdf = torch.where(times > theta, cdf_1, cdf_2)
        return cdf.clamp(min=1e-8, max=1.0)

    @torch.no_grad()
    def predict_survival(self, outputs, times):
        cdf = self.predict_cdf(outputs, times)
        return (1.0 - cdf).clamp(min=1e-8, max=1.0)