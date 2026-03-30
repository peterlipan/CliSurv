import os
from pathlib import Path
import torch
import warnings
import numpy as np
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
import time
from thop import profile
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import seaborn as sns


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
        self.plot_link_function()

    def train(self):
        args = self.args
        self.model.train()
        cur_iters = 0
        epoch_times = []
        epoch_mem_usage = []
        
        for i in range(args.epochs):
            epoch_start_time = time.time()
            torch.cuda.reset_peak_memory_stats()
            
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
            
            # epoch_time in seconds
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            epoch_mem_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # Convert to MB
        
        self.avg_epoch_time = np.mean(epoch_times)
        self.avg_gpu_memory = np.mean(epoch_mem_usage)

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
        if args.method.lower() in ['deepsurv', 'lassocox', 'coxtime', 'ald', 'lognorm', 'icald']:
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

            _, nll_rc = discrete_rc_nll(events=event_indicator, labels=labels, surv_bins=surv_prob, pad_left=args.pad_left, pad_right=args.pad_right)

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
            metric_dict['Epoch_Time'] = self.avg_epoch_time
            metric_dict['GPU_Memory_MB'] = self.avg_gpu_memory
            
            # Calculate number of parameters
            n_params = sum(p.numel() for p in self.model.parameters())
            metric_dict['Parameters'] = n_params
            
            # Calculate FLOPs using a sample batch
            # Calculate FLOPs using a sample batch
            try:
                sample_data = next(iter(self.test_loader))
                sample_data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in sample_data.items()}

                macs, _ = profile(self.model, inputs=(sample_data,), verbose=False)
                metric_dict['FLOPs'] = macs
            except Exception as e:
                warnings.warn(f"Could not compute FLOPs: {e}")
        
        self.model.train(training)

        return metric_dict

    def plot_link_function(self):
        """
        Render the learned inverse-link comparison at ICML publication quality.
    
        Reads from self.args and self.model; saves PNG (+ optional PDF) to
        <args.results>/links/<dataset>_<fold>.{png,pdf}.
        """
        rcParams.update({
            "text.usetex":         False,          # flip to True when LaTeX is present
            "font.family":         "serif",
            "font.serif":          ["Times New Roman", "DejaVu Serif", "Palatino"],
            "mathtext.fontset":    "stix",
            "axes.labelsize":      11,
            "axes.titlesize":      10,
            "xtick.labelsize":     9,
            "ytick.labelsize":     9,
            "legend.fontsize":     9,
            "legend.title_fontsize": 8.5,
            "legend.framealpha":   0.93,
            "legend.edgecolor":    "#d0d0d0",
            "legend.handlelength": 2.6,
            "lines.linewidth":     1.6,
            "axes.linewidth":      0.75,
            "xtick.major.width":   0.75,
            "ytick.major.width":   0.75,
            "xtick.minor.width":   0.45,
            "ytick.minor.width":   0.45,
            "xtick.major.size":    4.0,
            "ytick.major.size":    4.0,
            "xtick.minor.size":    2.2,
            "ytick.minor.size":    2.2,
            "xtick.direction":     "in",
            "ytick.direction":     "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "figure.dpi":          200,
            "savefig.dpi":         300,
            "savefig.bbox":        "tight",
        })

        _PAL = {
            "gen": "#0072B2",    # blue        – CliSurv-Gen (focal)
            "po":  "#E69F00",    # amber       – PO reference
            "ph":  "#D55E00",    # vermillion  – PH reference
            "ann": "#555555",    # annotation grey
            "ref": "#bbbbbb",    # asymptote / grid
        }
 

        args    = self.args
        img_dir = os.path.join(args.results, "links")
        os.makedirs(img_dir, exist_ok=True)
        stem    = os.path.join(img_dir, f"{args.dataset}")
    
        method = args.method.lower()
        if not method.startswith("clisurv"):
            return
    
        link = method.split("-")[1] if "-" in method else ""
        if link != "gen":
            return
    
        # ── Data ─────────────────────────────────────────────────────────────────
        info = self.model.activation.export_link_curves(num_points=1000)
        z    = np.asarray(info["z"])
        gen  = np.asarray(info["gen_invlink"])
        po   = np.asarray(info["po_invlink"])
        ph   = np.asarray(info["ph_invlink"])
    
        # ── Seaborn theme base (then we override below) ───────────────────────────
        sns.set_theme(style="ticks", context="paper", font="serif",
                    rc={"axes.spines.top": False, "axes.spines.right": False})
    
        fig, ax = plt.subplots(figsize=(5.2, 3.9))
    
        # ── Confidence-band shading for CliSurv-Gen (visual weight anchor) ────────
        ax.fill_between(z, gen - 0.015, gen + 0.015,
                        color=_PAL["gen"], alpha=0.10, linewidth=0, zorder=1)
    
        # ── Reference curves (drawn beneath focal) ────────────────────────────────
        ax.plot(z, po, color=_PAL["po"],
                linewidth=1.5, linestyle=(0, (5, 2.2)),
                label=r"PO  (logistic)",   zorder=2, alpha=0.90)
        ax.plot(z, ph, color=_PAL["ph"],
                linewidth=1.5, linestyle=(0, (2, 1.6)),
                label=r"PH  (clog-log)",   zorder=2, alpha=0.90)
    
        # ── Focal learned curve ───────────────────────────────────────────────────
        ax.plot(z, gen, color=_PAL["gen"],
                linewidth=2.5, linestyle="-",
                label=r"CliSurv-Gen",      zorder=3)
    
        # ── Asymptote reference lines ─────────────────────────────────────────────
        for yv in (0.0, 1.0):
            ax.axhline(yv, color=_PAL["ref"], linewidth=0.6,
                    linestyle=":", zorder=0)
    
        # ── Axes limits ───────────────────────────────────────────────────────────
        xpad = (z.max() - z.min()) * 0.015
        ax.set_xlim(z.min() - xpad, z.max() + xpad)
        ax.set_ylim(-0.05, 1.05)
    
        # ── Labels ────────────────────────────────────────────────────────────────
        ax.set_xlabel(r"Latent score $z$")
        ax.set_ylabel(r"Inverse link $g^{-1}(z)$")
    
        # ── Tick locators ─────────────────────────────────────────────────────────
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    
        # ── Horizontal grid only (seaborn ticks style already removed right/top) ──
        ax.yaxis.grid(True, linestyle="--", linewidth=0.45,
                    color=_PAL["ref"], alpha=0.55, zorder=0)
        ax.set_axisbelow(True)
    
        # ── Probability annotations (position is data-range-relative) ─────────────
        xspan = z.max() - z.min()
        x_lo  = z.min() + xspan * 0.03
        x_hi  = z.max() - xspan * 0.03
        _ann  = dict(fontsize=8, color=_PAL["ann"], fontstyle="italic",
                    va="bottom", linespacing=1.35)
        ax.text(x_lo, 0.035, "Lower event\nprobability", ha="left",  **_ann)
        ax.text(x_hi, 0.835, "Higher event\nprobability", ha="right", **_ann)
    
        _arw = dict(arrowstyle="-|>", color=_PAL["ann"],
                    lw=0.75, mutation_scale=7)
        ax.annotate("", xy=(x_lo + xspan * 0.005, 0.16),
                    xytext=(x_lo + xspan * 0.005, 0.26), arrowprops=_arw)
        ax.annotate("", xy=(x_hi - xspan * 0.005, 0.82),
                    xytext=(x_hi - xspan * 0.005, 0.72), arrowprops=_arw)
    
        # ── Legend ────────────────────────────────────────────────────────────────
        leg = ax.legend(loc="lower right", frameon=True,
                        borderpad=0.55, labelspacing=0.30,
                        handletextpad=0.5)
        leg.get_frame().set_linewidth(0.55)
    
        # ── Seaborn despine (ensures clean look after theme override) ─────────────
        sns.despine(ax=ax, top=True, right=True)
    
        # ── Export ────────────────────────────────────────────────────────────────
        fig.tight_layout(pad=0.35)
        fig.savefig(stem + ".png", dpi=300)
        # fig.savefig(stem + ".pdf")          # vector PDF for paper submission
        plt.show()
        plt.close(fig)
    
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
        args.n_bins = self.train_dataset.n_classes

        settings = ['Dataset', 'Method', 'Model', 'KFold', 'Epochs', 'Seed', 'Step (days)', 'Train Ratio', 'Layers', 'Hidden Dim', 'Activation', 'N_bins', 'Ranking Weight', 'Z_m']
        kwargs = ['dataset','method', 'backbone', 'kfold', 'epochs', 'seed', 'step', 'train_ratio', 'n_layers', 'd_hid', 'activation', 'n_bins', 'w_rank', 'z_m']

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
