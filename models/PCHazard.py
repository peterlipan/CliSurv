import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
from .utils import ModelOutputs
from pycox.models.loss import nll_pc_hazard_loss


class PCHazardLoss(nn.Module):
    def __init__(self):
        super().__init__()    

    def forward(self, outputs, data):
        return nll_pc_hazard_loss(
            outputs.logits,
            data['label'],
            data['event'],
            outputs.frac
        )


class PCHazard(nn.Module):
    def __init__(self, args):
        super(PCHazard, self).__init__()

        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes
        self.head = nn.Linear(self.d_hid, self.n_classes)
        self.criterion = PCHazardLoss()
        self.pad_left = args.pad_left
        self.step = args.step
    
    def forward(self, data):
        features = self.encoder(data['data'])
        logits = self.head(features)

        hazards = F.softplus(logits)
        cum_hazard = torch.cumsum(hazards, dim=1)
        surv = torch.exp(-cum_hazard)
        risk = - torch.sum(surv, dim=1)

        durations = data['duration']
        time_idx = data['label']
        t_0 = (time_idx - self.pad_left) * self.step
        frac = (durations - t_0) / self.step
        # optional: eps + clamp here

        return ModelOutputs(
            features=features,
            logits=logits,
            hazards=hazards,
            surv=surv,
            risk=risk,
            frac=frac,
        )
    
    def compute_loss(self, outputs, data):
        return self.criterion(
            outputs,
            data
        )
