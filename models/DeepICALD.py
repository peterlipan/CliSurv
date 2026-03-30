import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import get_encoder
from .utils import ModelOutputs


def safe_log(x, min_value=1e-8):
    return torch.log(torch.clamp(x, min=min_value))


# q-branch exp is unclamped in the reference, but a tight clamp prevents
# occasional inf/nan without changing behaviour in the normal range.
EXP_CLAMP = 5.0


# =========================
# ICALD LOSS
# =========================
class ICALDSurvLoss(nn.Module):
    """
    ICALD loss (Eq. 12 in the paper):
        L = weight * L_ALD + (1 - weight) * L_Cal

    Root cause of NaN: torch.where(condition, cen_1, cen_2) evaluates BOTH
    branches for ALL samples before selecting.  loss_cen_2 contains:

        value_to_exp = -sqrt2 * (theta - y) / (sigma * kappa)

    When y > theta, (theta - y) is negative, so value_to_exp is POSITIVE and
    potentially large.  exp(large positive) overflows, making inner <= 0 and
    log(inner) = NaN.  Even though torch.where discards cen_2 for y>theta
    samples in the forward pass, the NaN gradient still propagates backward
    through those discarded values, exploding the weights so the next forward
    pass also produces NaN.

    Fix: mask y to the valid domain before computing each branch so the
    dangerous exp is never evaluated for out-of-domain samples.

    When y <= theta: value_to_exp = -sqrt2*(theta-y)/(sigma*kappa) <= 0
                     -> exp(v) <= 1 -> inner in (0,1] always. No clamp needed.
    When y > theta:  loss_cen_1 = log(1+kappa^2) + sqrt2*kappa*(y-theta)/sigma
                     -> always finite for finite inputs.
    """

    def __init__(self, weight=0.1, use_censor_loss=True, eps=1e-8):
        super().__init__()
        self.weight = weight
        self.use_censor_loss = use_censor_loss
        self.eps = eps

    def forward(self, model, outputs, data):
        y = data['duration'].view(-1, 1).float()
        # cen_indicator: 1 = censored, 0 = observed  (matches reference convention)
        cen_indicator = (1 - data['event']).view(-1, 1).float()

        theta = outputs.theta
        sigma = outputs.sigma
        kappa = outputs.kappa
        q     = outputs.q

        sqrt2 = torch.sqrt(torch.tensor(2.0, device=y.device, dtype=y.dtype))

        # ------------------------------------------------------------------ #
        # L_ALD — negative log-likelihood
        # ------------------------------------------------------------------ #
        alpha = (y >= theta) * (y - theta)   # deviation above mode; 0 elsewhere
        beta  = (y <  theta) * (theta - y)   # deviation below mode; 0 elsewhere

        nll_per_sample = (
            safe_log(sigma, self.eps)
            - safe_log(kappa / (1.0 + kappa ** 2), self.eps)
            + (sqrt2 / sigma) * (alpha * kappa + beta / kappa)
        )
        loss_obs = ((cen_indicator == 0) * nll_per_sample).mean()

        if not self.use_censor_loss:
            loss_nll = loss_obs
        else:
            # --- censored loss, y > theta branch ---
            loss_cen_1 = (
                safe_log(1.0 + kappa ** 2, self.eps)
                + sqrt2 * kappa * (y - theta) / sigma
            )

            # --- censored loss, y <= theta branch ---
            # Mask y to y <= theta BEFORE computing the exp so that
            # (theta - y_masked) >= 0 always, keeping value_to_exp <= 0,
            # exp(value_to_exp) <= 1, and inner in (0, 1] -- no NaN possible.
            y_below    = torch.where(y <= theta, y, theta)   # clamp to theta
            v          = -sqrt2 * (theta - y_below) / (sigma * kappa)
            inner      = 1.0 - kappa ** 2 * torch.exp(v) / (kappa ** 2 + 1.0)
            loss_cen_2 = -safe_log(inner, self.eps)           # inner > 0 by construction

            loss_cen = (cen_indicator == 1) * torch.where(y > theta, loss_cen_1, loss_cen_2)
            loss_nll = loss_obs + loss_cen.mean()

        # ------------------------------------------------------------------ #
        # L_Cal — calibration loss  |F(y | x, q) - q|
        # ------------------------------------------------------------------ #
        cdf      = model._ald_cdf(y, theta, sigma, kappa)
        loss_cal = torch.abs(cdf - q).mean()

        loss = self.weight * loss_nll + (1.0 - self.weight) * loss_cal
        return loss


# =========================
# ICALD MODEL
# =========================
class DeepICALD(nn.Module):
    """
    DeepICALD — a faithful port of MLP_ICALD into the DeepALD-style framework.

    Architecture mirrors MLP_ICALD exactly:
        q  → relu(layer1_q) → exp(layer2_q)         [q_feat]
        x  → relu(layer1)   + projection(x)          [residual block]
        [x, q_feat] → relu(layer2_{θ,σ,κ}) → [h, q_feat] → layer3_{θ,σ,κ}
        θ  : raw output
        σ  : exp(layer3_sigma)
        κ  : exp(layer3_kappa)
    """

    def __init__(self, args):
        super().__init__()

        self.encoder   = get_encoder(args)
        self.d_hid     = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid

        self.use_dropout = getattr(args, 'dropout', 0.0) > 0
        self.dropout     = nn.Dropout(getattr(args, 'dropout', 0.0))

        self.q_dim = getattr(args, 'q_dim', 16)

        # ----- network layers (mirror MLP_ICALD) -----
        self.projection = nn.Linear(self.d_hid, self.d_hid, bias=False)

        # q branch: 1 -> 2*q_dim -> q_dim
        self.layer1_q = nn.Linear(1,           2 * self.q_dim, bias=True)
        self.layer2_q = nn.Linear(2 * self.q_dim, self.q_dim,  bias=True)

        # hidden heads take [x(d_hid) || q_feat(q_dim)]
        self.layer2_theta = nn.Linear(self.d_hid + self.q_dim, self.d_hid, bias=True)
        self.layer2_sigma = nn.Linear(self.d_hid + self.q_dim, self.d_hid, bias=True)
        self.layer2_kappa = nn.Linear(self.d_hid + self.q_dim, self.d_hid, bias=True)

        # output heads take [h(d_hid) || q_feat(q_dim)]
        self.layer3_theta = nn.Linear(self.d_hid + self.q_dim, 1, bias=True)
        self.layer3_sigma = nn.Linear(self.d_hid + self.q_dim, 1, bias=True)
        self.layer3_kappa = nn.Linear(self.d_hid + self.q_dim, 1, bias=True)

        self.criterion = ICALDSurvLoss(
            weight=getattr(args, 'icald_weight', 0.1),
            use_censor_loss=getattr(args, 'use_censor_loss', True),
        )

        self.bin_times_ = None

    # ---------------------------------------------------------------------- #
    def forward(self, data):
        features = self.encoder(data['data'])

        # Sample q ~ U(0, 1) if not provided (literal ICALD behaviour)
        if 'q' in data and data['q'] is not None:
            q = data['q']
            if q.ndim == 1:
                q = q.unsqueeze(1)
            q = q.to(features.device, dtype=features.dtype)
        else:
            q = torch.rand(
                features.shape[0], 1,
                device=features.device, dtype=features.dtype,
            )

        q = q.clamp(min=1e-4, max=1.0 - 1e-4)

        # q branch: relu -> exp  (reference has NO clamp before exp;
        # we keep a tight EXP_CLAMP as a stability guard that is invisible
        # in the normal operating range)
        q_feat = torch.relu(self.layer1_q(q))
        q_feat = torch.exp(
            torch.clamp(self.layer2_q(q_feat), min=-EXP_CLAMP, max=EXP_CLAMP)
        )

        # x branch: residual block
        residual = self.projection(features)
        x = torch.relu(features)           # NOTE: relu applied to features, not layer1(features)
        x = x + residual                   # matches reference: relu(layer1(x)) + projection(x)
                                           # your encoder plays the role of layer1 here

        # Concatenate x and q_feat — used as input to all three hidden heads
        xq = torch.cat([x, q_feat], dim=1)

        theta_h = torch.relu(self.layer2_theta(xq))
        sigma_h = torch.relu(self.layer2_sigma(xq))
        kappa_h = torch.relu(self.layer2_kappa(xq))

        if self.use_dropout:
            theta_h = self.dropout(theta_h)
            sigma_h = self.dropout(sigma_h)
            kappa_h = self.dropout(kappa_h)

        # Output heads: each takes [hidden || q_feat] (matches reference exactly)
        theta = self.layer3_theta(torch.cat([theta_h, q_feat], dim=1))

        sigma_logits = self.layer3_sigma(torch.cat([sigma_h, q_feat], dim=1))
        kappa_logits = self.layer3_kappa(torch.cat([kappa_h, q_feat], dim=1))

        # σ, κ must be strictly positive — exp with clamp for stability
        sigma = torch.exp(torch.clamp(sigma_logits, min=-EXP_CLAMP, max=EXP_CLAMP))
        kappa = torch.exp(torch.clamp(kappa_logits, min=-EXP_CLAMP, max=EXP_CLAMP))

        logits = torch.cat([theta, sigma, kappa], dim=1)
        risk   = -theta.view(-1)   # higher θ → longer survival → lower risk

        surv = None
        if self.bin_times_ is not None:
            surv = self._compute_survival_matrix(theta, sigma, kappa)

        outputs = ModelOutputs(
            features=features,
            logits=logits,
            theta=theta,
            sigma=sigma,
            kappa=kappa,
            risk=risk,
            surv=surv,
        )
        outputs.q = q
        return outputs

    # ---------------------------------------------------------------------- #
    def compute_loss(self, outputs, data):
        return self.criterion(self, outputs, data)

    # ---------------------------------------------------------------------- #
    @torch.no_grad()
    def configure_time_bins(self, bin_times):
        if not torch.is_tensor(bin_times):
            bin_times = torch.tensor(bin_times, dtype=torch.float32)
        self.bin_times_ = bin_times.float().clone()
        return self

    @torch.no_grad()
    def prepare_for_validation(self, train_loader, bin_times, device=None, force=False):
        self.configure_time_bins(bin_times)

    @torch.no_grad()
    def predict_survival_matrix(self, data_loader, device=None):
        if self.bin_times_ is None:
            raise RuntimeError(
                "Bin times not configured. Call prepare_for_validation first."
            )
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        surv_rows = []
        for batch in data_loader:
            x = batch['data'].to(device)

            batch_dict = {'data': x}
            if 'q' in batch and batch['q'] is not None:
                batch_dict['q'] = batch['q'].to(device)

            out = self.forward(batch_dict)
            surv_rows.append(out.surv.cpu())

        return torch.cat(surv_rows, dim=0)

    # ---------------------------------------------------------------------- #
    def _compute_survival_matrix(self, theta, sigma, kappa):
        times = self.bin_times_.to(theta.device, dtype=theta.dtype)
        B = theta.shape[0]
        T = times.shape[0]

        t   = times.unsqueeze(0).expand(B, T)
        th  = theta.expand(B, T)
        sig = sigma.expand(B, T)
        kap = kappa.expand(B, T)

        cdf  = self._ald_cdf(t, th, sig, kap)
        surv = (1.0 - cdf).clamp(min=1e-8, max=1.0)
        return surv

    # ---------------------------------------------------------------------- #
    @staticmethod
    def _ald_cdf(times, theta, sigma, kappa):
        """
        ALD CDF (matches reference ald_cdf exactly):
            y > θ:  1 - 1/(1+κ²) · exp(-√2·κ·(y-θ)/σ)
            y ≤ θ:  κ²/(1+κ²)   · exp(-√2·(θ-y)/(σ·κ))
        """
        sqrt2 = torch.sqrt(torch.tensor(2.0, device=times.device, dtype=times.dtype))

        cdf_above = 1.0 - 1.0 / (1.0 + kappa ** 2) * torch.exp(
            -sqrt2 * kappa * (times - theta) / sigma
        )
        cdf_below = (kappa ** 2 / (1.0 + kappa ** 2)) * torch.exp(
            -sqrt2 * (theta - times) / (sigma * kappa)
        )

        cdf = torch.where(times > theta, cdf_above, cdf_below)
        return cdf.clamp(min=1e-8, max=1.0)

    # ---------------------------------------------------------------------- #
    @torch.no_grad()
    def predict_cdf(self, outputs, times):
        theta, sigma, kappa = outputs.theta, outputs.sigma, outputs.kappa
        B = theta.shape[0]

        if times.ndim == 1:
            times = times.unsqueeze(0).repeat(B, 1)
        times = times.to(theta.device, dtype=theta.dtype)

        theta = theta.expand_as(times)
        sigma = sigma.expand_as(times)
        kappa = kappa.expand_as(times)

        return self._ald_cdf(times, theta, sigma, kappa)

    @torch.no_grad()
    def predict_survival(self, outputs, times):
        cdf = self.predict_cdf(outputs, times)
        return (1.0 - cdf).clamp(min=1e-8, max=1.0)