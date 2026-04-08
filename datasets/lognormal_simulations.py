import os
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Data generation (ported from SyntheticLogNormalDataset)
# ---------------------------------------------------------------------------

class _LogNormalGenerator:
    def __init__(self, input_dim: int, x_range: tuple, rng: np.random.Generator, sigma: float = 1.0):
        self.input_dim = input_dim
        self.x_range = x_range
        self.rng = rng
        self.sigma = float(sigma)

    def _mu(self, x: np.ndarray) -> np.ndarray:
        mu = (x - 1.0) ** 2
        return mu.mean(axis=1) if x.ndim == 2 else mu

    def _sigma(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0] if x.ndim == 2 else 1
        return np.full(n, self.sigma, dtype=np.float32)

    def sample_x(self, n: int) -> np.ndarray:
        return self.rng.uniform(self.x_range[0], self.x_range[1],
                                size=(n, self.input_dim)).astype(np.float32)

    def sample_event_times(self, x: np.ndarray) -> np.ndarray:
        return self.rng.lognormal(mean=self._mu(x), sigma=self._sigma(x)).astype(np.float32)

    def sample_censor_times(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        return self.rng.uniform(0.0, 10.0, size=n).astype(np.float32)


# ---------------------------------------------------------------------------
# Public dataset class
# ---------------------------------------------------------------------------

class LogNormalDataset:
    """
    Synthetic LogNormal survival dataset with the same public interface as
    ``SimSACConstTrainTestData``.

    CSV schema
    ----------
    - covariates : x0 … x{input_dim-1}
    - duration   : observed time  y = min(T, C)  (raw, no unit conversion)
    - event      : 1 = observed event, 0 = censored

    Parameters
    ----------
    root : str
        Directory for cached CSV files.
    n_train : int
        Number of training samples.
    n_test : int
        Number of test samples.
    input_dim : int
        Covariate dimension (default 1).
    x_range : tuple
        Range for uniform covariate sampling (default (0.0, 2.0)).
    normalize : bool
        Whether to z-score normalise covariates (default False).
    step : float
        Bin width for duration discretisation (default 1.0).
    seed : int
        Random seed (default 42).
    pad_left : int
        Label offset added on the left (default 1).
    pad_right : int
        Extra classes added on the right (default 1).
    is_censor_train : bool
        Apply censoring to the training set (default True).
    is_censor_test : bool
        Apply censoring to the test set (default False).
    """

    def __init__(
        self,
        root: str,
        n_train: int = 4000,
        n_test: int = 1000,
        input_dim: int = 8,
        sigma: float = 0.3,
        x_range: tuple = (0.0, 2.0),
        normalize: bool = False,
        step: float = 1.0,
        seed: int = 42,
        pad_left: int = 1,
        pad_right: int = 1,
        is_censor_train: bool = True,
        is_censor_test: bool = True,
    ):
        self.root = root
        self.n_train = int(n_train)
        self.n_test = int(n_test)
        self.input_dim = int(input_dim)
        self.x_range = tuple(x_range)
        self.normalize = normalize
        self.step = float(step)
        self.seed = int(seed)
        self.pad_left = int(pad_left)
        self.pad_right = int(pad_right)
        self.is_censor_train = is_censor_train
        self.is_censor_test = is_censor_test
        self.sigma = float(sigma)

        os.makedirs(self.root, exist_ok=True)

        # Unique cache filenames encode all generation parameters
        tag = (
            f"ntrain{self.n_train}_ntest{self.n_test}"
            f"_dim{self.input_dim}_seed{self.seed}"
            f"_cent{int(is_censor_train)}_cente{int(is_censor_test)}"
        )
        self.train_path = os.path.join(self.root, f"train_{tag}.csv")
        self.test_path = os.path.join(self.root, f"test_{tag}.csv")

        # Generate & cache if needed
        if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
            self._generate_and_save()

        # Load
        self.train_df = self._load_csv(self.train_path)
        self.test_df = self._load_csv(self.test_path)

        # Arrays
        self.train_data, self.train_duration, self.train_event = \
            self._df_to_arrays(self.train_df, normalize=self.normalize)
        self.test_data, self.test_duration, self.test_event = \
            self._df_to_arrays(self.test_df, normalize=self.normalize)

        # Dataset meta-properties
        self.n_features = self.train_data.shape[1]

        # n_events: exclude the censoring class (0), matching SimSACConstTrainTestData
        self.n_events = int(
            max(
                len(np.unique(self.train_event)) - 1,
                len(np.unique(self.test_event)) - 1,
            )
        )

        # Discretise durations
        self.train_label = self._duration_to_label(self.train_duration)
        self.test_label = self._duration_to_label(self.test_duration)

        max_label = max(self.train_label.max(), self.test_label.max())
        self.n_classes = int(max_label + self.pad_left + self.pad_right)

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------
    def _feature_cols(self):
        return [f"x{i}" for i in range(self.input_dim)]

    def _required_cols(self):
        return self._feature_cols() + ["duration", "event"]

    def _validate_df(self, df: pd.DataFrame, context: str):
        missing = set(self._required_cols()) - set(df.columns)
        if missing:
            raise ValueError(f"[{context}] Missing columns: {sorted(missing)}")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate_and_save(self):
        rng = np.random.default_rng(self.seed)
        gen = _LogNormalGenerator(self.input_dim, self.x_range, rng, sigma=self.sigma)

        n_total = self.n_train + self.n_test
        x_all = gen.sample_x(n_total)

        # --- training split ---
        x_train = x_all[: self.n_train]
        target_train = 200 * gen.sample_event_times(x_train)
        cen_train = (
            200 * gen.sample_censor_times(x_train)
            if self.is_censor_train
            else target_train + 100.0
        )
        y_train, event_train = self._apply_censoring(target_train, cen_train)

        # --- test split ---
        x_test = x_all[self.n_train:]
        target_test = 200 * gen.sample_event_times(x_test)
        cen_test = (
            200 * gen.sample_censor_times(x_test)
            if self.is_censor_test
            else target_test + 100.0
        )
        y_test, event_test = self._apply_censoring(target_test, cen_test)

        # --- save ---
        self._arrays_to_csv(x_train, y_train, event_train, self.train_path)
        self._arrays_to_csv(x_test, y_test, event_test, self.test_path)

    @staticmethod
    def _apply_censoring(target: np.ndarray, cen: np.ndarray):
        """Return observed times and event indicators (1=event, 0=censored)."""
        y = np.minimum(target, cen).astype(np.float32)
        event = (cen >= target).astype(np.int64)   # 1 if observed, 0 if censored
        return y, event

    def _arrays_to_csv(self, x, duration, event, path):
        df = pd.DataFrame(x, columns=self._feature_cols())
        df["duration"] = duration.astype(np.float32)
        df["event"] = event.astype(np.int64)
        df.to_csv(path, index=False)

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    def _load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        self._validate_df(df, context=f"load:{os.path.basename(path)}")
        df = df.copy()
        for c in self._feature_cols():
            df[c] = df[c].astype(np.float32)
        df["duration"] = df["duration"].astype(np.float32)
        df["event"] = df["event"].astype(np.int64)
        return df

    def _df_to_arrays(self, df: pd.DataFrame, normalize: bool = False):
        X = df[self._feature_cols()].values.astype(np.float32)
        duration = df["duration"].values.astype(np.float32)
        event = df["event"].values.astype(np.float32)
        if normalize:
            X = self._normalize(X)
        return X, duration, event

    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0
        return (X - mean) / std

    # ------------------------------------------------------------------
    # Discretisation
    # ------------------------------------------------------------------
    def _duration_to_label(self, duration: np.ndarray) -> np.ndarray:
        return (duration // self.step).astype(np.int64) + self.pad_left

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_official_train_test(self):
        """Return (train_dataset, test_dataset) as SurvivalDataset instances."""
        train_dataset = SurvivalDataset(
            parent_data=self,
            data=self.train_data,
            duration=self.train_duration,
            event=self.train_event,
            label=self.train_label,
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_events=self.n_events,
        )
        test_dataset = SurvivalDataset(
            parent_data=self,
            data=self.test_data,
            duration=self.test_duration,
            event=self.test_event,
            label=self.test_label,
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_events=self.n_events,
        )
        return train_dataset, test_dataset


# ---------------------------------------------------------------------------
# SurvivalDataset  (identical to SimSAC version)
# ---------------------------------------------------------------------------

class SurvivalDataset(Dataset):
    def __init__(self, parent_data, data, duration, event, label,
                 n_features, n_classes, n_events):
        self.data = data.astype(np.float32)
        self.duration = duration.astype(np.float32)
        self.event = event.astype(np.int64)
        self.label = label.astype(np.int64)
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_events = n_events
        self._duration_to_label = parent_data._duration_to_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "data": self.data[idx],
            "label": self.label[idx],
            "duration": self.duration[idx],
            "event": self.event[idx],
        }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = LogNormalDataset(
        root="./lognormal_cache",
        n_train=5_000,
        n_test=1_000,
        input_dim=1,
        step=0.5,
        seed=42,
    )
    train_ds, test_ds = loader.get_official_train_test()

    print(f"Training set size : {len(train_ds)}")
    print(f"Test set size     : {len(test_ds)}")
    print(f"Number of features: {train_ds.n_features}")
    print(f"Number of classes : {train_ds.n_classes}")
    print(f"Number of events  : {train_ds.n_events}")

    sample = train_ds[0]
    print(f"Sample keys       : {list(sample.keys())}")
    print(f"Sample data shape : {sample['data'].shape}")