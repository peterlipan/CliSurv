import os
from pathlib import Path
import torch
import warnings
import pandas as pd
from sksurv.util import Surv
from models import CreateModel
from datasets import CreateDataset
from .metrics import compute_surv_metrics, discrete_rc_nll
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from .utils import MetricLogger


class Trainer:
    def __init__(self, args, wb_logger=None, val_steps=None):
        self.args = args
        self.wb_logger = wb_logger
        self.val_steps = val_steps
        self.verbose = args.verbose
        self.m_logger = MetricLogger(n_folds=args.kfold)
        os.makedirs(args.checkpoints, exist_ok=True)
        os.makedirs(args.results, exist_ok=True)
    
    def _init_components(self):
        args = self.args
        print(f"Train dataset size: {len(self.train_dataset)}, Test dataset size: {len(self.test_dataset)}")
        print(f"Max duration in train set: {self.train_dataset.duration.max()}, Max duration in test set: {self.test_dataset.duration.max()}")
        print(f"Min duration in train set: {self.train_dataset.duration.min()}, Min duration in test set: {self.test_dataset.duration.min()}")

        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, 
                                       shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, 
                                      shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)
        
        args.n_classes = self.train_dataset.n_classes
        args.n_features = self.train_dataset.n_features

        self.model = CreateModel(args).cuda()

        self.optimizer = getattr(torch.optim, args.optimizer)(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if self.val_steps is None:
            self.val_steps = len(self.train_loader) 
        
        self.scheduler = None
        if args.lr_policy == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs * len(self.train_loader), eta_min=1e-6)
        elif args.lr_policy == 'cosine_restarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5)
        elif args.lr_policy == 'multi_step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[18, 19], gamma=0.1)  
        
    def official_train_test(self):
        # non-kfold training
        args = self.args
        dataset = CreateDataset(args)
        self.train_dataset, self.test_dataset = dataset.get_official_train_test()
        self._init_components()
        self.fold = 0
        
        self.train()
        metric_dict = self.validate()
        self.m_logger.update(metric_dict)
        print('-'*20, 'Metrics', '-'*20)
        print(metric_dict)
        self._save_fold_surv_avg_results(metric_dict)

    def kfold_train(self):
        args = self.args
        dataset = CreateDataset(args)

        for fold, (train_dataset, test_dataset) in enumerate(dataset.get_kfold_datasets()):
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
            self.fold = fold

            self._init_components()

            self.train()

            # validate for the fold
            metric_dict = self.validate()
            self.m_logger.update(metric_dict, fold)
            print('-'*20, f'Fold {fold} Metrics', '-'*20)
            print(metric_dict)

        avg_metrics = self.m_logger.fold_average()
        print('-'*20, 'Average Metrics', '-'*20)
        print(avg_metrics)
        self._save_fold_surv_avg_results(avg_metrics)

    def run(self):
        args = self.args
        if args.kfold > 1:
            self.kfold_train()
        else:
            self.official_train_test()

    def train(self):
        args = self.args
        self.model.train()
        cur_iters = 0
        for i in range(args.epochs):
            for data in self.train_loader:
                data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}

                outputs = self.model(data)
                loss = self.model.compute_loss(outputs, data)
                print(f"\rFold {self.fold} | Epoch {i} | Iter {cur_iters} | Loss: {loss.item()}", end='', flush=True)

                self.optimizer.zero_grad()
                loss.backward()

                # clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                cur_iters += 1
                if self.verbose:
                    if cur_iters % self.val_steps == 0:

                        cur_lr = self.optimizer.param_groups[0]['lr']
                        metric_dict = self.validate()
                        print('\n', '-'*20, 'Metrics', '-'*20)
                        for key, value in metric_dict.items():
                            print(f"{key}: {value}")
                        if self.wb_logger is not None:
                            self.wb_logger.log({f"Fold_{self.fold}": {
                                'Train': {'loss': loss.item(), 'lr': cur_lr},
                                'Test': metric_dict
                            }})

    def validate(self):
        args = self.args
        training = self.model.training
        self.model.eval()

        loss = 0.0
            
        event_indicator = torch.Tensor().cuda() # whether the event (death) has occurred
        duration = torch.Tensor().cuda()
        labels = torch.Tensor().cuda()
        risk_prob = torch.Tensor().cuda()
        surv_prob = torch.Tensor().cuda()

        # calculate the baseline_surv for deepsurv
        if args.method.lower() in ['deepsurv', 'lassocox', 'coxtime']:
            bin_times = (torch.arange(self.train_dataset.n_classes, dtype=torch.float32) - args.pad_left) * args.step
            bin_times = torch.clamp(bin_times, min=0.0)
            self.model.prepare_for_validation(self.train_loader, bin_times.to(duration.device))
                

        # for estimating censoring distribution in the training set
        train_duration = self.train_dataset.duration
        train_event = self.train_dataset.event.astype(bool)
        train_surv = Surv.from_arrays(event=train_event, time=train_duration)

        with torch.no_grad():
            for data in self.test_loader:
                data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}
                outputs = self.model(data)
                batch_loss = self.model.compute_loss(outputs, data)
                loss += batch_loss.item()
                
                risk = outputs.risk
                event_indicator = torch.cat((event_indicator, data['event']), dim=0)
                duration = torch.cat((duration, data['duration']), dim=0)
                risk_prob = torch.cat((risk_prob, risk), dim=0)
                surv_prob = torch.cat((surv_prob, outputs.surv), dim=0)
                labels = torch.cat((labels, data['label']), dim=0)

            event_indicator = event_indicator.cpu().detach().numpy().astype(bool)
            duration = duration.cpu().detach().numpy()
            risk_prob = risk_prob.cpu().detach().numpy()
            surv_prob = surv_prob.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            test_surv = Surv.from_arrays(event=event_indicator, time=duration)

            _, nll_rc = discrete_rc_nll(events=event_indicator, labels=labels, surv_bins=surv_prob)

            # Earliest event
            min_time = duration[event_indicator].min()
            # Last evaluable time = largest event time strictly less than max censoring
            censor_mask = ~event_indicator
            if censor_mask.any():
                max_censor_time = duration[censor_mask].max()
                max_time = duration[(event_indicator) & (duration < max_censor_time)].max()
            else:
                max_time = duration.max()  # no censoring case

            # Sample times within [min_time, max_time]
            N_eval = 1000 
            time_points = np.linspace(min_time, max_time, N_eval, endpoint=False, dtype=float)
            time_labels = self.test_dataset._duration_to_label(time_points)
            surv_prob_eval = surv_prob[:, time_labels]

            metric_dict = compute_surv_metrics(train_surv, test_surv, risk_prob, surv_prob_eval, time_points)
            metric_dict['NLL'] = nll_rc
            metric_dict['Loss'] = loss / len(self.test_loader)
        
        self.model.train(training)

        return metric_dict
    
    def save_model(self):
        args = self.args
        model_name = f"{args.method}_{args.backbone}.pt"
        save_path = os.path.join(args.checkpoints, model_name)
        torch.save(self.model.state_dict(), save_path)

    def _save_fold_surv_avg_results(self, metric_dict, keep_best=True):
        # keep_best: whether save the best model (highest mcc) for each fold
        args = self.args
        suffix = "_ours" if 'clisurv' in args.method.lower() else "_baseline"
        df_name = f"{args.kfold}Fold_{args.dataset}{suffix}.xlsx"
        res_path = args.results

        settings = ['Dataset', 'Method', 'Model', 'KFold', 'Epochs', 'Seed', 'Step (days)', 'Train Ratio', 'Layers', 'Hidden Dim', 'Activation']
        kwargs = ['dataset','method', 'backbone', 'kfold', 'epochs', 'seed', 'step', 'train_ratio', 'n_layers', 'd_hid', 'activation']

        # for simulations
        if hasattr(args, 'n_train'):
            settings.append('N_train')
            kwargs.append('n_train')
        if hasattr(args, 'n_test'):
            settings.append('N_test')
            kwargs.append('n_test')
        if hasattr(args, 'link'):
            settings.append('Link')
            kwargs.append('link')

        set2kwargs = {k: v for k, v in zip(settings, kwargs )}

        metric_names = self.m_logger.metrics()
        df_columns = settings + metric_names
        
        df_path = os.path.join(res_path, df_name)
        if not os.path.exists(df_path):
            df = pd.DataFrame(columns=df_columns)
        else:
            df = pd.read_excel(df_path)
            if df_columns != df.columns.tolist():
                warnings.warn("Columns in the existing excel file do not match the current settings.")
                df = pd.DataFrame(columns=df_columns)
        
        new_row = {k: args.__dict__[v] for k, v in set2kwargs.items()}

        if keep_best: # keep the rows with the best mcc for each fold
            reference = 'C-index'
            exsiting_rows = df[(df[settings] == pd.Series(new_row)).all(axis=1)]
            if not exsiting_rows.empty:
                exsiting_mcc = exsiting_rows[reference].values
                if metric_dict[reference] > exsiting_mcc:
                    df = df.drop(exsiting_rows.index)
                else:
                    return

        new_row.update(metric_dict)
        df = df._append(new_row, ignore_index=True)
        df.to_excel(df_path, index=False)
