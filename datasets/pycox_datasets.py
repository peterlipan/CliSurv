import torch
import pycox
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from pycox.datasets import metabric, support, gbsg, flchain, nwtco, sac3, rr_nl_nhp, sac_admin5


class PycoxDataset:
    def __init__(self, pycox_dataloader, step=1.0, stratify=False, 
                 kfold=5, seed=42, normalize=False, unit_scale=1.0, pad_left=1, pad_right=1, train_ratio=1.0):
        self.pycox_dataloader = pycox_dataloader
        self.df = self._load_df()
        self.data, self.duration, self.event = self._preprocess_data(normalize=normalize)
        self.n_features = self.data.shape[1]
        self.n_events = int(len(np.unique(self.event)) - 1)  # ignore censoring
        self.stratify = stratify
        self.kfold = kfold
        self.seed = seed
        self.step = step
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.unit_scale = unit_scale
        self.train_ratio = train_ratio

        # rescale the duration to unit of days
        self.duration = (self.duration * unit_scale).astype(np.float32)
        self.label = self._duration_to_label(self.duration)
        self.n_classes = int(self.label.max() + self.pad_left + self.pad_right)  # including padding at both ends

    def _load_df(self):
        return self.pycox_dataloader.read_df()
    
    def _preprocess_data(self, normalize=False):
        duration = self.df['duration'].values.astype(float)
        event = self.df['event'].values.astype(int)
        data = self.df.iloc[:, :-2].values.astype(float) # Exclude 'duration' and 'event' columns
        if normalize:
            data = self._normalize(data)
        return data, duration, event

    def _normalize(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1
        return (X - mean) / std

    def _duration_to_label(self, duration):
        bin_idx = (duration // self.step).astype(np.int64) + self.pad_left  # padding on the left
        return bin_idx
    
    def get_kfold_datasets(self):
        if self.stratify:
            kf = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)
            for train_idx, test_idx in kf.split(self.data, self.event):
                yield PytorchDataset(self, train_idx, ratio=self.train_ratio), PytorchDataset(self, test_idx, ratio=1.0)
        else:
            kf = KFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)
            for train_idx, test_idx in kf.split(self.data):
                yield PytorchDataset(self, train_idx, ratio=self.train_ratio), PytorchDataset(self, test_idx, ratio=1.0)

                
class PytorchDataset(Dataset):
    def __init__(self, pycox_data, indices, ratio=1.0):
        duration = pycox_data.duration[indices]
        event = pycox_data.event[indices]
        data = pycox_data.data[indices]
        label = pycox_data.label[indices]

        if ratio < 1.0:
            data, _, label, _, duration, _, event, _ = train_test_split(
                data, label, duration, event, train_size=ratio, random_state=pycox_data.seed, stratify=event
            )
        
        self.data = data
        self.label = label
        self.event = event
        self.duration = duration

        self.n_features = pycox_data.n_features
        self.n_classes = pycox_data.n_classes
        self.n_events = pycox_data.n_events
        self._duration_to_label = pycox_data._duration_to_label


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data': torch.tensor(self.data[idx], dtype=torch.float32),
            'label': torch.tensor(self.label[idx], dtype=torch.long),
            'event': torch.tensor(self.event[idx], dtype=torch.long),
            'duration': torch.tensor(self.duration[idx], dtype=torch.float32)
        }

# this raw data needs to be cleaned. Use deephit's instead.
class MetabricDataset(PycoxDataset):
    # the durations are in the unit of months
    def __init__(self, step=1., stratify=False, kfold=5, seed=42, normalize=False, pad_left=1, pad_right=1, train_ratio=1.0):
        super().__init__(metabric, step=step, stratify=stratify, kfold=kfold, seed=seed, normalize=normalize, unit_scale=30.4375, pad_left=pad_left, pad_right=pad_right, train_ratio=train_ratio)

class SupportDataset(PycoxDataset):
    def __init__(self, step=1., stratify=False, kfold=5, seed=42, normalize=False, pad_left=1, pad_right=1, train_ratio=1.0):
        super().__init__(support, step=step, stratify=stratify, kfold=kfold, seed=seed, normalize=normalize, unit_scale=1.0, pad_left=pad_left, pad_right=pad_right, train_ratio=train_ratio)

class GBSGDataset(PycoxDataset):
    def __init__(self, step=1., stratify=False, kfold=5, seed=42, normalize=False, pad_left=1, pad_right=1, train_ratio=1.0):
        super().__init__(gbsg, step=step, stratify=stratify, kfold=kfold, seed=seed, normalize=normalize, unit_scale=30.4375, pad_left=pad_left, pad_right=pad_right, train_ratio=train_ratio)

class FlchainDataset(PycoxDataset):
    def __init__(self, step=1., stratify=False, kfold=5, seed=42, normalize=False, pad_left=1, pad_right=1, train_ratio=1.0):
        super().__init__(flchain, step=step, stratify=stratify, kfold=kfold, seed=seed, normalize=normalize, unit_scale=1.0, pad_left=pad_left, pad_right=pad_right, train_ratio=train_ratio)

    def _load_df(self):
        df = self.pycox_dataloader.read_df(processed=False)
        # drop the categorical columns, sample.yr and flc.grp
        df = df.drop(['chapter', 'sample.yr', 'flc.grp'], axis=1).loc[lambda x: x['creatinine'].isna() == False].reset_index(drop=True).assign(sex=lambda x: (x['sex'] == 'M'))
        df = df.rename(columns={'futime': 'duration', 'death': 'event'})
        return df

class NWTCODataSet(PycoxDataset):
    def __init__(self, step=1., stratify=False, kfold=5, seed=42, normalize=False, pad_left=1, pad_right=1, train_ratio=1.0):
        super().__init__(nwtco, step=step, stratify=stratify, kfold=kfold, seed=seed, normalize=normalize, unit_scale=1.0, pad_left=pad_left, pad_right=pad_right, train_ratio=train_ratio)

    def _load_df(self):
        df = self.pycox_dataloader.read_df(processed=False)
        df = df.assign(instit_2=df['instit'] - 1, histol_2=df['histol'] - 1, study_4=df['study'] - 3,
                       stage=df['stage'].astype('category')).drop(['seqno', 'instit', 'histol', 'study', 'in.subcohort', 'rownames'], axis=1)
        df = df.rename(columns={'edrel': 'duration', 'rel': 'event'})
        # rearrange columns
        cols = ['duration', 'event']
        df = df[list(df.columns.drop(cols)) + cols]
        return df