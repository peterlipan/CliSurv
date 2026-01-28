import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import get_encoder
from .utils import ModelOutputs


class CQRNNLoss(nn.Module):
    """
    Drop-in replacement loss, parallel to NllSurvLoss.
    Uses label (interval index) and event indicator,
    exactly like LogisticHazards.
    """
    def __init__(self, n_classes, y_max=99.0,
                 use_censor_loss=True,
                 use_cross_loss=True,
                 margin=0.1, alpha_cross=10.0):
        super().__init__()

        # Hard-coded quantile grid
        # length must equal n_classes
        taus = torch.linspace(0.0, 1.0, steps=n_classes)
        self.register_buffer("taus_torch", taus)

        self.n_classes = n_classes
        self.y_max = y_max
        self.use_censor_loss = use_censor_loss
        self.use_cross_loss = use_cross_loss
        self.margin = margin
        self.alpha_cross = alpha_cross

    # ----- subcomponents from original CQRNN ----- #

    def _crossing_loss(self, y_pred):
        diffs = y_pred[:, 1:-1] - y_pred[:, :-2]
        loss_cross = self.alpha_cross * torch.mean(
            torch.maximum(torch.tensor(0.0, device=y_pred.device),
                          self.margin - diffs)
        )
        return loss_cross

    def _quantile_loss(self, y_pred, y_true, cen_indicator):
        tau_block = self.taus_torch.unsqueeze(0).repeat(y_pred.size(0), 1)
        loss = torch.sum(
            (cen_indicator < 1.0) *
            (y_pred - y_true) *
            ((1 - tau_block) - 1.0 * (y_pred < y_true)),
            dim=1
        )
        return loss.mean()

    def forward(self, outputs, data):
        """
        outputs.quantiles : (batch, n_classes)
        data['label']     : interval index (1...K)
        data['event']     : 1 = event, 0 = censored
        """
        y_pred = outputs.quantiles
        batch = y_pred.size(0)

        # Convert interval index to a representative continuous time
        # Here: midpoints of bins in normalized time [0,1]
        # You can change this if needed.
        label = data["label"].view(batch, 1).float()
        event = data["event"].view(batch, 1).float()
        cen_indicator = 1.0 - event

        # Convert interval index → time in [0,1]
        # Example: intervals are evenly spaced
        # t_k = k / n_classes
        y_true = (label - 1) / float(self.n_classes)

        # OBSERVED LOSS
        loss_obs = self._quantile_loss(y_pred, y_true, cen_indicator)

        if self.use_cross_loss:
            loss_obs += self._crossing_loss(y_pred)

        if not self.use_censor_loss:
            return loss_obs

        # ----- censored loss ----- #
        y_pred_detach = y_pred.detach()
        tau_block = self.taus_torch.unsqueeze(0).repeat(batch, 1)

        torch_abs = torch.abs(y_true - y_pred_detach[:, :-1])
        min_vals, _ = torch.min(torch_abs, dim=1, keepdim=True)
        closest_mask = (torch_abs == min_vals)
        est_q = torch.max(tau_block[:, :-1] * closest_mask, dim=1).values
        est_q = est_q.view(batch, 1)

        taus_no_last = tau_block[:, :-1]
        weights = (taus_no_last < est_q).float() + \
                  (taus_no_last >= est_q) * (taus_no_last - est_q) / (1.0 - est_q)

        y_max = self.y_max

        loss_cens = torch.sum(
            (cen_indicator > 0.0) *
            (
                weights *
                (y_pred[:, :-1] - y_true) *
                ((1.0 - taus_no_last) - 1.0 * (y_pred[:, :-1] < y_true))
                +
                (1.0 - weights) *
                (y_pred[:, :-1] - y_max) *
                ((1.0 - taus_no_last) - 1.0 * (y_pred[:, :-1] < y_max))
            ),
            dim=1
        )

        loss_cens = loss_cens.mean()

        return loss_obs + loss_cens


class CQRNN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes   # number of bins = number of quantile outputs

        # The same discretization used by METABRICData
        K = self.n_classes
        step = args.step         # MUST be added to args
        pad_left = args.pad_left # MUST be added to args

        time_grid = (torch.arange(K) - pad_left) * step
        self.register_buffer("time_grid", time_grid)

        # Quantile grid (hard-coded)
        taus = torch.linspace(0, 1, steps=K)
        self.register_buffer("taus", taus)

        self.head = nn.Linear(self.d_hid, self.n_classes)
        self.criterion = CQRNNLoss(n_classes=self.n_classes)

    @staticmethod
    def quantiles_to_surv(quantiles, taus, time_grid):
        """
        quantiles : (B, Q) predicted times Q_x(τ_q)
        taus      : (Q,)   quantile levels in [0,1]
        time_grid : (K,)   times at which to eval survival (e.g. bin midpoints)

        returns:
        surv : (B, K) survival probabilities at time_grid
        """
        B, Q = quantiles.shape
        K = time_grid.shape[0]
        device = quantiles.device

        surv = torch.zeros(B, K, device=device)
        eps = 1e-8

        for k in range(K):
            t = time_grid[k]

            # mask of quantiles >= t
            mask = (quantiles >= t)  # (B, Q)

            # default: if no quantile >= t, set τ_est ≈ 1.0 (CDF ~ 1, survival ~ 0)
            tau_est = torch.ones(B, device=device)

            # if at least one quantile >= t, get the first such index
            any_ge = mask.any(dim=1)
            if any_ge.any():
                idx_first = torch.argmax(mask[any_ge].float(), dim=1)  # first True

                # gather Q_x(τ_i), τ_i
                q_i = quantiles[any_ge, :].gather(1, idx_first.view(-1, 1)).squeeze(1)
                tau_i = taus[idx_first]

                # crude: F(t) ≈ τ_i; you can add interpolation if you want
                tau_est[any_ge] = tau_i

            # S(t) = 1 - F(t)
            surv[:, k] = 1.0 - tau_est

        return surv


    def forward(self, data):
        features = self.encoder(data["data"])
        quantiles = self.head(features)

        # Convert quantiles → survival probability on discrete time grid
        surv = self.quantiles_to_surv(quantiles, self.taus, self.time_grid)

        risk = -surv.sum(dim=1)

        return ModelOutputs(
            features=features,
            logits=quantiles,  # raw
            quantiles=quantiles,
            surv=surv,
            risk=risk,
        )
    
    def compute_loss(self, outputs, data):
        return self.criterion(outputs, data)