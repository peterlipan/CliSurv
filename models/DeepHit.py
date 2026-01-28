import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
from .utils import ModelOutputs
from pycox.models.loss import DeepHitSingleLoss


class DeepHitsurvLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dhl = DeepHitSingleLoss(alpha=0.5, sigma=0.5)
    
    @staticmethod
    def pair_rank_mat(idx_durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        dur_i = idx_durations.view(-1, 1)
        dur_j = idx_durations.view(1, -1)
        ev_i = events.view(-1, 1)
        ev_j = events.view(1, -1)

        rank_mat = ((dur_i < dur_j) | ((dur_i == dur_j) & (ev_j == 0))).float() * ev_i
        return rank_mat

    def forward(self, outputs, data):
        # directly predict the risk factors
        rank_mat = self.pair_rank_mat(data['label'], data['event'])
        return self.dhl(outputs.logits, data['label'], data['event'], rank_mat)


class DeepHit(nn.Module):
    def __init__(self, args):
        super(DeepHit, self).__init__()
        
        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes
        self.head = nn.Linear(self.d_hid, args.n_classes)
        self.n_classes = args.n_classes
        self.criterion = DeepHitsurvLoss()
    
    def forward(self, data):
        features = self.encoder(data['data'])
        logits = self.head(features)
        pmf = F.softmax(logits, dim=1)  # [B, K]
        
        # expected first hitting time (risk score for C-index)
        time_bins = torch.arange(1, pmf.size(1)+1, device=pmf.device).float()
        risk = -(pmf * time_bins).sum(dim=1)  # higher risk -> earlier expected event
        
        # cumulative
        cdf = torch.cumsum(pmf, dim=1)
        surv = 1.0 - cdf
        
        # optional: discrete argmax FHT
        fht = torch.argmax(pmf, dim=1)
        prob_at_fht = torch.gather(pmf, 1, fht.unsqueeze(1)).squeeze(1)
        
        return ModelOutputs(features=features,
                            logits=logits,
                            pmf=pmf,
                            risk=risk,
                            cdf=cdf,
                            surv=surv,
                            fht=fht,
                            prob_at_fht=prob_at_fht)


    def compute_loss(self, outputs, data):
        return self.criterion(
            outputs,
            data
        )
