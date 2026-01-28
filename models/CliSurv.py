import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
from .utils import ModelOutputs


class MonotoneISplineLink(nn.Module):
    """
    General cumulative-link transformation using I-spline-like basis:
        h(z) = sigmoid( alpha * z + beta * sum_m softplus(w_m) * I_m(z) )

    - I_m(z) are non-decreasing basis functions, precomputed on a grid.
    - softplus(w_m) >= 0 ensures the spline part is monotone in z.
    - alpha, beta control how close we are to plain logistic (PO).
      At init: alpha ~ 1, beta ~ 0 -> behaves like logit.
    """
    def __init__(
        self,
        num_basis: int = 8,
        grid_size: int = 200,
        z_min: float = -8.0,
        z_max: float = 8.0,
        eps: float = 1e-6
    ):
        super().__init__()
        self.num_basis = num_basis
        self.grid_size = grid_size
        self.z_min = z_min
        self.z_max = z_max

        # Fixed grid over logits
        z_grid = torch.linspace(z_min, z_max, grid_size)  # [G]
        self.register_buffer("z_grid", z_grid)

        # Knot positions for hat (degree-1 B-spline) basis
        knots = torch.linspace(z_min, z_max, num_basis)
        self.register_buffer("knots", knots)  # [M]

        # Precompute I-spline-like basis I[g, m] on the grid
        I_grid = self._precompute_ispline_basis(z_grid, knots)  # [G, M]
        self.register_buffer("I_grid", I_grid)

        # Learnable spline weights (non-negative via softplus)
        self.raw_weights = nn.Parameter(torch.full((num_basis,), -8.0))
        # softplus(-4) ~ 0.018 -> very small contribution at init

        # Learnable scales for the linear (PO) and spline parts
        self.alpha = nn.Parameter(torch.tensor(1. - eps))   # starts as logit
        self.beta  = nn.Parameter(torch.tensor(0. + eps))   # no spline at init

        # Optional global bias (can be 0)
        self.bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _hat_basis(z: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
        G = z.shape[0]
        M = knots.shape[0]
        z_exp = z.view(G, 1)
        k_exp = knots.view(1, M)

        d = torch.abs(z_exp - k_exp)
        if M > 1:
            dx = (knots[1] - knots[0]).item()
        else:
            dx = 1.0
        H = torch.clamp(1.0 - d / dx, min=0.0)

        # Normalize rowwise so sum_m H_m(z) ≈ 1
        H_sum = H.sum(dim=1, keepdim=True) + 1e-8
        H = H / H_sum
        return H

    def _precompute_ispline_basis(self, z_grid: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
        H = self._hat_basis(z_grid, knots)          # [G, M]
        dz = z_grid[1] - z_grid[0]
        I = torch.cumsum(H * dz, dim=0)             # [G, M]

        # normalize each basis to [0, 1]
        I_max = I[-1, :].clone()
        I_max[I_max <= 0] = 1.0
        I = I / I_max.view(1, -1)
        return I

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Clamp logits to grid range
        z_clamped = z.clamp(self.z_min, self.z_max)

        orig_shape = z_clamped.shape
        z_flat = z_clamped.view(-1)  # [N]

        # Map z to grid indices
        G = self.grid_size
        z_min, z_max = self.z_min, self.z_max
        u = (z_flat - z_min) / (z_max - z_min + 1e-8) * (G - 1)
        u_clamped = u.clamp(0, G - 1 - 1e-6)

        idx0 = u_clamped.floor().long()
        idx1 = torch.clamp(idx0 + 1, max=G - 1)
        w1 = (u_clamped - idx0.float()).unsqueeze(-1)  # [N, 1]

        I0 = self.I_grid[idx0]    # [N, M]
        I1 = self.I_grid[idx1]    # [N, M]
        I_z = (1.0 - w1) * I0 + w1 * I1  # [N, M]

        # Non-negative spline weights, initially very small
        w_pos = F.softplus(self.raw_weights)  # [M] >= 0

        spline_term = torch.matmul(I_z, w_pos)  # [N]

        # Combined link: linear (logit) + spline deviation
        g_flat = self.bias + self.alpha * z_flat + self.beta * spline_term

        h_flat = torch.sigmoid(g_flat)
        return h_flat.view(orig_shape)



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
        exponent = (-diff_risk / self.sigma).clamp(max=30) # prevent overflow
        loss = rank_mat * torch.exp(exponent)
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
        return nll + self.rank_weight * rloss
    

class CliSurv(nn.Module):
    """
    Cumulative-link survival model with link-specific, monotone baseline
    parameterization.

    - PH: learns cumulative baseline hazard Λ₀(t)
    - PO: learns baseline CDF F₀(t) directly (no identifiable hazard object)
    - Gen: learns baseline CDF passed through a flexible monotone link

    The baseline is parameterized via positive increments to ensure monotonicity
    and stable optimization.
    """

    def __init__(self, args, link: str = "po", eps: float = 1e-4):
        super().__init__()

        # -------------------------
        # Backbone & head
        # -------------------------
        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid
        self.n_classes = int(args.n_classes)

        self.head = nn.Linear(self.d_hid, 1, bias=False)
        self.criterion = CDFLoss()

        self.link = link.lower()
        self.eps = float(eps)

        if self.link == "gen":
            self.activation = MonotoneISplineLink()

        # Baseline reparameterization
        self.raw_deltas = nn.Parameter(torch.full((self.n_classes,), -3.0))
        self.log_scale = nn.Parameter(torch.tensor(1.0))
        self._cached_baseline_logits = None

    def _baseline_logits(self) -> torch.Tensor:
        eps = self.eps

        # positive increments -> monotone u_k in (0, 1]
        delta = F.softplus(self.raw_deltas) + eps          # [T] > 0
        u = torch.cumsum(delta, dim=0)                     # [T]
        u = u / (u[-1] + eps)                              # normalize to (0,1]

        if self.link == "ph":
            # Proportional Hazards:
            #   Λ₀(k) = scale * u_k
            #   b_k = log Λ₀(k)
            scale = torch.exp(self.log_scale)
            Lambda = scale * u
            b = torch.log(Lambda + eps)

        elif self.link in {"po", "gen"}:
            # Proportional Odds / General:
            # Learn baseline CDF directly (natural object under NLL)
            F0 = eps + (1.0 - 2.0 * eps) * u
            F0 = F0.clamp(eps, 1.0 - eps)
            b = torch.log(F0 / (1.0 - F0))                 # logit(F₀)

        else:
            raise ValueError(f"Unknown link: {self.link}")

        return b

    # -------------------------------------------------
    # Link activation
    # -------------------------------------------------
    def activate(self, logits: torch.Tensor) -> torch.Tensor:
        if self.link == "po":
            return torch.sigmoid(logits)

        if self.link == "ph":
            # cloglog
            return 1.0 - torch.exp(-torch.exp(logits))

        if self.link == "pro":
            normal = torch.distributions.Normal(0.0, 1.0)
            return normal.cdf(logits)

        if self.link == "gen":
            return self.activation(logits)

        raise ValueError(f"Unknown link function: {self.link}")

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self, data):
        x = data["data"]
        features = self.encoder(x)
        proj = self.head(features)                          # [B, 1]

        # baseline logits
        b = self._baseline_logits()                         # [T]
        self._cached_baseline_logits = b

        logits = proj + b.view(1, -1)                       # [B, T]
        cdf = self.activate(logits)                         # [B, T]
        risk = proj.view(-1)                                # [B]

        # enforce monotonicity only at evaluation
        if not self.training:
            cdf = torch.cummax(cdf, dim=1).values
            cdf = cdf.clamp_(min=self.eps, max=1.0 - self.eps)

        surv = 1.0 - cdf

        return ModelOutputs(
            features=features,
            logits=logits,
            cdf=cdf,
            risk=risk,
            surv=surv,
        )

    def compute_loss(self, outputs, data):
        return self.criterion(outputs, data)

    # -------------------------------------------------
    # Baseline export (for analysis / figures)
    # -------------------------------------------------
    @torch.no_grad()
    def save_baseline(self, ground_truth_survival: np.ndarray):
        """
        Save baseline survival S₀(k) for proj = 0.

        Note:
        - Meaningful comparison to GT baseline is expected ONLY for PH.
        - For PO / Gen, this is provided for completeness and visualization.
        """
        ground_truth_survival = np.asarray(ground_truth_survival, dtype=np.float32)
        if ground_truth_survival.ndim != 1:
            raise ValueError("ground_truth_survival must be 1D")

        b = self._baseline_logits()
        F0 = self.activate(b)
        F0 = torch.cummax(F0, dim=0).values
        F0 = F0.clamp_(min=self.eps, max=1.0 - self.eps)

        S0 = (1.0 - F0).cpu().numpy().astype(np.float32)

        if S0.shape[0] != ground_truth_survival.shape[0]:
            raise ValueError(
                f"baseline length {S0.shape[0]} != gt length {ground_truth_survival.shape[0]}"
            )

        return {
            "method": f"CliSurv-{self.link}",
            "baseline_survival": S0,
            "ground_truth_survival": ground_truth_survival,
        }
