import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import get_encoder
from .utils import ModelOutputs


class NMTLRLoss(nn.Module):
    """
    N-MTLR negative log-likelihood, as in Fotso (2018):

        L(θ) = - Σ_i [ (1-δ_i) log S(t_i | x_i) + δ_i log f(t_i | x_i) ]

    where:
      - f_k(x) = P(T in interval k | x)  (pmf over discrete time intervals)
      - S_k(x) = P(T >= left boundary of interval k | x)
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    @staticmethod
    def _prep_targets(idx_duration, event):
        batch_size = idx_duration.size(0)
        idx_duration = idx_duration.view(batch_size, 1).long()
        event = event.view(batch_size, 1).float()
        return idx_duration, event

    def forward(self, outputs, data):
        f = outputs.hazards      # (B, K)  pmf: P(T in interval k | x)
        S = outputs.surv         # (B, K)  survival at left boundary of interval k
        idx = data['label']
        event = data['event']

        idx, event = self._prep_targets(idx, event)
        eps = self.eps

        # gather f_s and S_s at the observed interval index s
        f_obs = torch.gather(f, 1, idx).clamp(min=eps)     # P(T in interval s | x)
        S_obs = torch.gather(S, 1, idx).clamp(min=eps)     # P(T >= left boundary of s | x)

        # uncensored: log f_s
        loglik_event = event * torch.log(f_obs)
        # censored: log S_s
        loglik_cens = (1.0 - event) * torch.log(S_obs)

        loglik = loglik_event + loglik_cens
        nll = -loglik.mean()

        return nll


class NMTLR(nn.Module):
    """
    Neural Multi-Task Logistic Regression (N-MTLR)
    """

    def __init__(self, args, eps: float = 1e-7):
        super().__init__()
        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes  # K intervals
        self.eps = eps

        self.head = nn.Linear(self.d_hid, self.n_classes)

        self.criterion = NMTLRLoss(eps=eps)

    def forward(self, data):

        x = data['data']
        features = self.encoder(x)           # (B, d_hid)
        logits = self.head(features)         # (B, K)  = z_k(x)

        psi = torch.flip(torch.cumsum(torch.flip(logits, dims=[1]), dim=1), dims=[1])

        # pmf over intervals: f_k(x) = P(T in interval k | x)
        f = F.softmax(psi, dim=1)           # (B, K), Σ_k f_k = 1

        S_rev = torch.cumsum(torch.flip(f, dims=[1]), dim=1)
        S = torch.flip(S_rev, dims=[1])     # (B, K)
        S = torch.clamp(S, min=self.eps)

        risk = -S.sum(dim=1)

        return ModelOutputs(
            features=features,
            logits=logits,
            hazards=f,   # pmf
            surv=S,      # survival at left boundary
            risk=risk,
        )

    def compute_loss(self, outputs, data):
        return self.criterion(outputs, data)
