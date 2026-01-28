import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
from .utils import ModelOutputs


class CDFLoss(nn.Module):
    def __init__(self, sigma: float = 0.5, rank_weight: float = 0.5, mono_weight: float = 0.5,
                 margin: float = 0.3):
        super().__init__()
        self.sigma = sigma
        self.rank_weight = rank_weight
        self.mono_weight = mono_weight
        self.margin = margin          # additive margin

    @staticmethod
    def pair_rank_mat(idx_durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        dur_i = idx_durations.view(-1, 1)
        dur_j = idx_durations.view(1, -1)
        ev_i = events.view(-1, 1)
        ev_j = events.view(1, -1)
        return ((dur_i < dur_j) | ((dur_i == dur_j) & (ev_j == 0))).float() * ev_i

    def rank_loss_on_risk(self, risk: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        si = risk.view(-1, 1)
        sj = risk.view(1, -1)
        diff_risk = si - sj - self.margin
        rank_mat = self.pair_rank_mat(durations, events)
        loss = rank_mat * torch.exp(-diff_risk / self.sigma)
        return loss.sum() / (rank_mat.sum() + 1e-6)

    @staticmethod
    def nll_from_cdf(cdf, label, event, eps=1e-7):
        B, T = cdf.shape
        prevF = torch.cat([cdf.new_zeros(B, 1), cdf[:, :-1]], dim=1)
        pmf = (cdf - prevF).clamp_min(eps)        # p_k >= 0
        surv_tail = (1.0 - cdf).clamp_min(eps)    # S(y) >= 0
        idx = torch.arange(B, device=cdf.device)

        py = pmf[idx, label].clamp_min(eps)       # p_y
        Sy = surv_tail[idx, label]                # S(y)

        loss_e = -torch.log(py[event == 1]).mean() if (event == 1).any() else 0.0
        loss_c = -torch.log(Sy[event == 0]).mean() if (event == 0).any() else 0.0
        return loss_e + loss_c
    
    def monotonicity_loss(self, cdf: torch.Tensor) -> torch.Tensor:
        diffs = cdf[:, 1:] - cdf[:, :-1]
        violations = F.relu(-diffs)
        return violations.mean()


    def forward(self, outputs, data):
        F_pred = outputs.cdf
        risk   = outputs.risk
        label  = data['label']
        event  = data['event']
        duration = data['duration']

        nll   = self.nll_from_cdf(F_pred, label, event)
        rloss = self.rank_loss_on_risk(risk, duration, event)
        mloss = self.monotonicity_loss(F_pred)
        return nll + self.rank_weight * rloss + self.mono_weight * mloss


class Decouple(nn.Module):
    def __init__(self, args):
        super(Decouple, self).__init__()

        self.encoder   = get_encoder(args)
        self.d_hid     = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes

        # Same linear head; we will normalize its weight for the angular score
        self.head   = nn.Linear(self.d_hid, 1, bias=False)

        self.biases = nn.Parameter(torch.linspace(-1, 1, self.n_classes), requires_grad=True)
        self.criterion = CDFLoss()

        # Global scale kept for backward compatibility; now multiplied with per-sample tau
        self.scaler = nn.Parameter(2. * torch.ones(1))

        # NEW: tiny temperature head mapping radius r = ||features|| to a positive tau
        # (softplus keeps it > 0; simple and minimal)
        self.temp = nn.Linear(1, 1)
        self._eps = 1e-7

    def _tau_from_radius(self, r: torch.Tensor) -> torch.Tensor:
        # r: [B,1]  -> tau: [B,1], strictly positive
        return F.softplus(self.temp(r)) + 1e-4

    def forward(self, data):
        features = self.encoder(data['data'])                # [B,D]

        # --- Decoupling starts here ---
        # Angular (scale-free) score for ranking
        r = features.norm(dim=1, keepdim=True).clamp_min(1e-6)  # [B,1]
        u = features / r                                         # [B,D] unit direction
        w_hat = F.normalize(self.head.weight, dim=1)             # [1,D]
        s_ang = (u @ w_hat.t())                                  # [B,1] in [-1,1]

        # Per-sample temperature for calibration (likelihood path)
        tau = self._tau_from_radius(r)                           # [B,1]

        # NLL logits: same angular location + learned thresholds, scaled by tau (and global scaler)
        logits = (s_ang + self.biases.view(1, -1)) * self.scaler * tau   # [B,T]
        cdf    = torch.sigmoid(logits)                                     # [B,T]
        # --- Decoupling ends here ---

        # force monotonicity only during eval
        if not self.training:
            cdf = torch.cummax(cdf, dim=1).values.clamp_(min=self._eps, max=1.0 - self._eps)

        surv = 1.0 - cdf
        risk = s_ang.view(-1)       # ranking uses angles only

        return ModelOutputs(features=features,
                            logits=logits,
                            cdf=cdf,
                            risk=risk,
                            surv=surv,
                            biases=self.biases,
                            projection_weight=self.head.weight.view(-1),
                            radius=r.view(-1),
                            tau=tau.view(-1))

    def compute_loss(self, outputs, data):
        return self.criterion(outputs, data)
