import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
from .utils import ModelOutputs
from pycox.models.loss import DeepHitSingleLoss


class NllSurvLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def nll_loss(hazards, surv_prob, idx_duration, event, alpha=0.4, eps=1e-7):
        batch_size = len(idx_duration)
        idx_duration = idx_duration.view(batch_size, 1) # ground truth bin, 1,2,...,k
        event = event.view(batch_size, 1).float()
        if surv_prob is None:
            surv_prob = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
        # without padding, S(0) = S[0], h(0) = h[0]
        surv_prob_padded = torch.cat([torch.ones_like(event), surv_prob], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
        # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
        #h[y] = h(1)
        #S[1] = S(1)
        uncensored_loss = -event * (torch.log(torch.gather(surv_prob_padded, 1, idx_duration).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, idx_duration).clamp(min=eps)))
        censored_loss = - (1 - event) * torch.log(torch.gather(surv_prob_padded, 1, idx_duration+1).clamp(min=eps))
        neg_l = censored_loss + uncensored_loss
        loss = (1-alpha) * neg_l + alpha * uncensored_loss
        loss = loss.mean()
        return loss

    def forward(self, outputs, data):
        return self.nll_loss(
            outputs.hazards,
            outputs.surv,
            data['label'],
            data['event'],
        )


class LogisticHazards(nn.Module):
    def __init__(self, args):
        super(LogisticHazards, self).__init__()

        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes
        self.head = nn.Linear(self.d_hid, self.n_classes)
        self.criterion = NllSurvLoss()
    
    def forward(self, data):
        features = self.encoder(data['data'])
        logits = self.head(features)
        hazards = F.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        risk = - torch.sum(surv, dim=1)  # risk is the negative cumulative survival
        return ModelOutputs(features=features, logits=logits, hazards=hazards, surv=surv, risk=risk)

    def compute_loss(self, outputs, data):
        return self.criterion(
            outputs,
            data
        )
