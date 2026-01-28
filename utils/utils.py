import numpy as np

class MetricLogger:
    def __init__(self, n_folds=None):
        """
        Metric logger compatible with both k-fold cross-validation and official train/test split
        
        Args:
            n_folds: Number of folds for k-fold CV. If None and is_kfold=True, defaults to 5
            is_kfold: Whether this is for k-fold CV (True) or official train/test (False)
        """
        is_kfold = n_folds > 0
        self.is_kfold = is_kfold
        
        if is_kfold:
            self.n_folds = n_folds if n_folds is not None else 5
            self.fold_metrics = [{} for _ in range(self.n_folds)]
        else:
            # For official train/test, we only need one "fold"
            self.n_folds = 1
            self.fold_metrics = [{}]

    def update(self, metric_dict, fold=0):
        """
        Update metrics for a specific fold
        
        Args:
            metric_dict: Dictionary of metrics
            fold: Fold number (ignored for official train/test)
        """
        if self.is_kfold:
            if fold >= self.n_folds:
                raise ValueError(f"Fold {fold} exceeds number of folds {self.n_folds}")
            self.fold_metrics[fold] = metric_dict
        else:
            # For official train/test, always update the single metrics dict
            self.fold_metrics[0] = metric_dict

    def metrics(self):
        """Get list of available metric names"""
        if self.fold_metrics[0]:
            return list(self.fold_metrics[0].keys())
        return []

    def fold_average(self):
        """
        Get average metrics across folds
        For official train/test, returns the single metric dict
        For k-fold, returns average across all folds
        """
        if not self.is_kfold:
            # For official train/test, return the single result
            return self.fold_metrics[0].copy()
        
        # For k-fold, calculate average across folds
        if not self.fold_metrics[0]:
            return {}
            
        avg_metrics = {k: 0.0 for k in self.metrics()}
        valid_folds = [fold for fold in self.fold_metrics if fold]  # Only non-empty folds
        
        if not valid_folds:
            return avg_metrics
            
        for metric in avg_metrics:
            metric_values = [fold[metric] for fold in valid_folds if metric in fold]
            if metric_values:
                avg_metrics[metric] = np.mean(metric_values)
        
        return avg_metrics

    def get_fold_metrics(self, fold=0):
        """Get metrics for a specific fold"""
        if fold >= self.n_folds:
            raise ValueError(f"Fold {fold} exceeds number of folds {self.n_folds}")
        return self.fold_metrics[fold].copy()

    def get_all_fold_metrics(self):
        """Get all fold metrics"""
        return self.fold_metrics.copy()

    def print_summary(self):
        """Print summary of metrics"""
        if self.is_kfold:
            print('-' * 20, 'K-Fold Summary', '-' * 20)
            for fold in range(self.n_folds):
                if self.fold_metrics[fold]:
                    print(f'Fold {fold}: {self.fold_metrics[fold]}')
            avg_metrics = self.fold_average()
            print('-' * 20, 'Average Metrics', '-' * 20)
            print(avg_metrics)
        else:
            print('-' * 20, 'Official Train/Test Results', '-' * 20)
            print(self.fold_metrics[0])