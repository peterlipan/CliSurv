import os
import json
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
        if args.method == 'deepcdf' and args.n_bins > 0:
            self._fold_plot_2d(metric_dict, self.test_loader, training_set=False)
            self._fold_plot_2d(metric_dict, self.train_loader, training_set=True)
        self._save_fold_surv_avg_results(metric_dict)
        # self.save_survival_predictions(self.test_loader, split="test")

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
            # self.save_survival_predictions(self.test_loader, split="oof")

            # self.fold_univariate_cox_regression_analysis(args, fold)

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
        
        if args.dataset.lower() == 'links':
            if args.method.lower() in ['clisurv-ph', 'clisurv-po', 'clisurv-gen', 'deepsurv', 'lassocox']:
                result_folder = os.path.join(args.results, "links", args.link)
                os.makedirs(result_folder, exist_ok=True)
                result_path = os.path.join(result_folder, f"{args.method}.npz")
                ground_truth_survival = self.train_dataset.survival_at_grids
                surv_dict = self.model.save_baseline(ground_truth_survival=ground_truth_survival)
                np.savez(result_path, **surv_dict)
                

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
    
    
    @torch.no_grad()
    def save_survival_predictions(self, dataloader, split: str = "test"):
        """
        Save per-patient survival S(t) at the model's *effective* (non-padded) bins.
        Appends across folds to a single Parquet; writes a JSON sidecar with exact
        effective bin endpoints (days) and padding info.

        Columns written:
          patient_id, event_time_days, event_indicator, Fold, Split, Dataset, Method, Backbone, Step_days,
          Pad_left, Pad_right, S_000 ... S_{T_eff-1}

        Notes:
        - durations in your datasets are already standardized to *days*.
        - padding defaults to 1 left + 1 right; override via args.pad_left/args.pad_right if needed.
        - works for all survival methods as long as forward() returns outputs.surv [B, T_native].
        """
        args = self.args
        was_training = self.model.training
        self.model.eval()

        # Cox-style models may need baseline prep (mirror your validate())
        if args.method.lower() in ['deepsurv', 'lassocox', 'coxtime']:
            # Use the training dataset's native "bin indices" (as in your validate())
            bin_times = torch.arange(self.train_dataset.n_classes, dtype=torch.float32)
            self.model.prepare_for_validation(self.train_loader, bin_times.cuda())

        # Padding config (your setup is 1+1; keep configurable)
        pad_l = args.pad_left
        pad_r = args.pad_right

        # Collect predictions
        all_pid, all_event, all_time = [], [], []
        blocks = []
        row_counter = 0
        T_native = None

        for batch in dataloader:
            batch_cuda = {k: (v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v) for k, v in batch.items()}
            outputs = self.model(batch_cuda)                        # expects outputs.surv: [B, T_native]
            S_native = outputs.surv.detach().cpu().numpy()          # [B, T_native]
            B, T = S_native.shape
            if T_native is None:
                T_native = T

            # Strip padding on both sides → effective bins only
            if pad_l + pad_r >= T:
                raise ValueError(f"Padding ({pad_l}+{pad_r}) >= T_native ({T}).")
            S_eff = S_native[:, pad_l:T - pad_r]                   # [B, T_eff]
            blocks.append(S_eff)

            # IDs (dataset may not provide patient_id/filename → synthesize fold-stable IDs)
            pids = batch.get('patient_id', batch.get('filename', None))
            if pids is None:
                pids = [f"Fold{getattr(self,'fold',0)}_row{row_counter+i}" for i in range(B)]
            else:
                if torch.is_tensor(pids):
                    pids = pids.detach().cpu().tolist()
                pids = [str(x) for x in pids]

            # durations/events are already *days*
            ev = batch['event']; du = batch['duration']
            if torch.is_tensor(ev): ev = ev.detach().cpu().numpy()
            if torch.is_tensor(du): du = du.detach().cpu().numpy()

            all_pid.extend(pids)
            all_event.extend(ev.astype(bool).tolist())
            all_time.extend(du.tolist())
            row_counter += B

        if T_native is None:
            self.model.train(was_training)
            return  # nothing to save

        S_mat = np.vstack(blocks)                # [N, T_eff]
        T_eff = S_mat.shape[1]

        # Effective right-endpoint times in *days* (padding removed).
        # Prefer dataset-provided bin times if available; else derive from step.
        if hasattr(self.train_dataset, "bin_times") and self.train_dataset.bin_times is not None:
            times_days_eff = np.asarray(self.train_dataset.bin_times, dtype=float)[pad_l:T_native - pad_r].tolist()
        else:
            step_days = float(getattr(args, "step", 1.0))
            times_days_eff = (np.arange(1, T_eff + 1, dtype=float) * step_days).tolist()

        # Build dataframe
        S_cols = [f"S_{k:03d}" for k in range(T_eff)]
        df = pd.DataFrame(S_mat, columns=S_cols)
        df.insert(0, "patient_id", all_pid)
        df.insert(1, "event_time_days", all_time)
        df.insert(2, "event_indicator", np.asarray(all_event, dtype=bool))
        df["Fold"] = int(getattr(self, "fold", 0))
        df["Split"] = split                           # "test" (external) or "oof" (kfold held-out)
        df["Dataset"] = args.dataset
        df["Method"] = args.method
        df["Backbone"] = args.backbone
        df["Step_days"] = float(getattr(args, "step", 1.0))
        df["Pad_left"] = pad_l
        df["Pad_right"] = pad_r

        # Paths
        base_dir = Path(args.results) / "surv_preds" / args.dataset
        base_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{args.method}__{args.backbone}__step{int(args.step)}d"
        parquet_path = base_dir / f"{stem}.parquet"
        sidecar_path = base_dir / f"{stem}.json"

        # Append across folds
        if parquet_path.exists():
            prev = pd.read_parquet(parquet_path)
            combined = pd.concat([prev, df], ignore_index=True)
            combined.drop_duplicates(subset=["patient_id", "Fold", "Split"], keep="last", inplace=True)
            combined.to_parquet(parquet_path, index=False)
        else:
            df.to_parquet(parquet_path, index=False)

        # JSON sidecar (exact effective bin endpoints, days)
        sidecar = {
            "model_name": f"{args.method}_{args.backbone}",
            "dataset": args.dataset,
            "discretization_step_days": float(getattr(args, "step", 1.0)),
            "times_days_eff": times_days_eff,            # effective right-endpoints (padding removed)
            "right_continuous": True,
            "n_bins_eff": int(T_eff),
            "pad_left": pad_l,
            "pad_right": pad_r,
            "horizon_days": float(times_days_eff[-1]) if len(times_days_eff) else 0.0,
        }
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f, indent=2)

        self.model.train(was_training)

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
        

    def fold_univariate_cox_regression_analysis(self):
        args = self.args
        fold = self.fold
        training = self.model.training
        self.model.eval()

        event_indicator = torch.empty(0).cuda()
        duration = torch.empty(0).cuda()
        risk_factor = torch.empty(0).cuda()
        filename = []
        patient_id = []

        df_name = f"{args.kfold}Fold_{args.dataset}_Cox.xlsx"
        res_path = args.results
        df_path = os.path.join(res_path, df_name)
        
                
        with torch.no_grad():
            for data in self.test_loader:
                data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}
                outputs = self.model(data)
                risk = outputs.risk
                event_indicator = torch.cat((event_indicator, data['event']), dim=0)
                duration = torch.cat((duration, data['duration']), dim=0)
                risk_factor = torch.cat((risk_factor, risk), dim=0)
                filename.extend(data['filename'])
                patient_id.extend(data['patient_id'])
        
        event_indicator = event_indicator.cpu().numpy()
        duration = duration.cpu().numpy()
        risk_factor = risk_factor.cpu().numpy()
                

        fold_df = pd.DataFrame({
            'BBNumber': patient_id,
            'Filename': filename,
            'Fold': [fold] * len(filename),
            'event': event_indicator,
            'duration': duration,
            f'{args.include}_{args.backbone}_{args.surv_loss}': risk_factor,
        })

        if hasattr(self, 'cox_df'):
            self.cox_df = pd.concat([self.cox_df, fold_df], ignore_index=True)
        else:
            self.cox_df = fold_df

        if fold == args.kfold - 1:
            if os.path.exists(df_path):
                existing_df = pd.read_excel(df_path)
                existing_df[f'{args.backbone}'] = None  # Initialize the new column

                for _, row in self.cox_df.iterrows():
                    filename = row['Filename']
                    if filename in existing_df['Filename'].values:
                        existing_df.loc[existing_df['Filename'] == filename, f'{args.include}_{args.backbone}_{args.surv_loss}'] = row[f'{args.include}_{args.backbone}_{args.surv_loss}']
            else:
                existing_df = self.cox_df
            existing_df.to_excel(df_path, index=False)

        self.model.train(training)       


