import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class RotterdamGBSGData:
    def __init__(self, step=1.0, seed=42, pad_left=1, pad_right=1, train_ratio=1.0):
        """
        Combined Rotterdam and GBSG dataset class
        Rotterdam is used as training set, GBSG as external validation
        
        Args:
            step: Time step for discretization (default: 1.0)
            seed: Random seed (default: 42)
        """
        self.rotterdam_data, self.rotterdam_duration, self.rotterdam_event = self._load_rotterdam()
        self.gbsg_data, self.gbsg_duration, self.gbsg_event = self._load_gbsg()
        
        # Use Rotterdam characteristics for dataset properties
        self.n_features = self.rotterdam_data.shape[1]
        self.n_events = int(max(len(np.unique(self.rotterdam_event)) - 1, 
                               len(np.unique(self.gbsg_event)) - 1))  # ignore censoring
        
        self.seed = seed
        self.step = step
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.train_ratio = train_ratio
        
        # Convert duration to labels
        self.rotterdam_label = self._duration_to_label(self.rotterdam_duration)
        self.gbsg_label = self._duration_to_label(self.gbsg_duration)
        
        # Calculate n_classes based on both datasets
        max_label = max(self.rotterdam_label.max(), self.gbsg_label.max())
        self.n_classes = int(max_label + self.pad_left + self.pad_right)  # including padding at both ends

    def _load_rotterdam(self):
        """Load and preprocess Rotterdam dataset"""
        # Load dataset
        try:
            rotterdam = pd.read_stata("http://www.stata-press.com/data/fpsaus/rott2.dta")
        except Exception as e:
            rotterdam = pd.read_stata("./datasets/RtmGBSG/rott2.dta")
            print("Loaded Rotterdam dataset from local file due to error:", e)
        
        # Select columns of interest
        rotterdam_cols = ["age", "meno", "size", "nodes", "er", "hormon", "rf", "rfi"]
        rotterdam = rotterdam[rotterdam_cols]
        
        # Rename columns
        rotterdam = rotterdam.rename(columns={
            "rf": "duration",
            "rfi": "event"
        })
        
        # Preprocess meno: map strings to 0/1. 0 -> pre, 1->post
        rotterdam["meno"] = rotterdam["meno"].map({"pre": 0, "post": 1}).astype(int)
        
        # Preprocess size: recode strings into categories
        size_map = {"<=20 mm": 0, ">20-50mmm": 1, ">50 mm": 2}
        rotterdam["size"] = rotterdam["size"].map(size_map).astype(int)
        
        # Preprocess hormon
        rotterdam["hormon"] = rotterdam["hormon"].map({"no": 0, "yes": 1}).astype(int)
        
        # Convert duration from months to days
        rotterdam["duration"] = 30.4375 * rotterdam["duration"]
        
        # Extract features and labels
        feature_cols = ["age", "meno", "size", "nodes", "er", "hormon"]
        data = rotterdam[feature_cols].values.astype(np.float32)
        duration = rotterdam["duration"].values.astype(np.float32)
        event = rotterdam["event"].values.astype(np.float32)
        
        return data, duration, event

    def _load_gbsg(self):
        """Load and preprocess GBSG dataset"""
        # Load dataset
        try:
            gbsg = pd.read_stata("http://www.stata-press.com/data/r11/brcancer.dta")
        except Exception as e:
            gbsg = pd.read_stata("./datasets/RtmGBSG/brcancer.dta")
            print("Loaded GBSG dataset from local file due to error:", e)
        
        # Select columns of interest
        gbsg_cols = ["x1", "x2", "x3", "x5", "x7", "hormon", "rectime", "censrec"]
        gbsg = gbsg[gbsg_cols]
        
        # Rename columns
        gbsg = gbsg.rename(columns={
            "x1": "age",
            "x2": "meno",
            "x3": "size",
            "x5": "nodes",
            "x7": "er",
            "rectime": "duration",
            "censrec": "event"
        })
        
        # Preprocess meno: recode {1,2} -> {0,1}
        gbsg["meno"] = gbsg["meno"].map({1: 0, 2: 1}).astype(int)
        
        # Preprocess size: bin into same categories as Rotterdam
        def recode_size(x):
            if x <= 20:
                return 0
            elif x <= 50:
                return 1
            else:
                return 2
        gbsg["size"] = gbsg["size"].apply(recode_size).astype(int)
        
        # Extract features and labels
        feature_cols = ["age", "meno", "size", "nodes", "er", "hormon"]
        data = gbsg[feature_cols].values.astype(np.float32)
        duration = gbsg["duration"].values.astype(np.float32)
        event = gbsg["event"].values.astype(np.float32)
        
        return data, duration, event

    def _duration_to_label(self, duration):
        """Convert duration to discrete labels"""
        bin_idx = (duration // self.step).astype(np.int64) + self.pad_left  # padding on the left
        return bin_idx

    def get_official_train_test(self):
        """
        Return official train/test split
        Rotterdam as training set, GBSG as external validation/test set
        """
        train_dataset = SurvivalDataset(
            parent_data=self,
            data=self.rotterdam_data,
            duration=self.rotterdam_duration,
            event=self.rotterdam_event,
            label=self.rotterdam_label,
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_events=self.n_events,
            ratio=self.train_ratio
        )
        
        test_dataset = SurvivalDataset(
            parent_data=self,
            data=self.gbsg_data,
            duration=self.gbsg_duration,
            event=self.gbsg_event,
            label=self.gbsg_label,
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_events=self.n_events,
            ratio=1.0
        )
        
        return train_dataset, test_dataset


class SurvivalDataset(Dataset):
    def __init__(self, parent_data, data, duration, event, label, n_features, n_classes, n_events, ratio=1.0):
        """
        Survival dataset for PyTorch DataLoader
        
        Args:
            parent_data: Reference to parent RotterdamGBSGData instance
            data: Feature data array
            duration: Duration/survival times
            event: Event indicators
            label: Discretized duration labels
            n_features: Number of features
            n_classes: Number of time bins
            n_events: Number of event types
        """
        if ratio < 1.0:
            data, _, duration, _, event, _, label, _ = train_test_split(
                data, duration, event, label,
                train_size=ratio,
                random_state=parent_data.seed,
                shuffle=True,
                stratify=event
            )
        self.data = data.astype(np.float32)
        self.duration = duration.astype(np.int64)
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
            'data': self.data[idx],
            'label': self.label[idx],
            'duration': self.duration[idx],
            'event': self.event[idx],
        }


# Usage example:
if __name__ == "__main__":
    # Initialize the data loader
    data_loader = RotterdamGBSGData(step=30.0)  # 30-day bins
    
    # Get train and test datasets
    train_dataset, test_dataset = data_loader.get_official_train_test()
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Number of features: {train_dataset.n_features}")
    print(f"Number of classes: {train_dataset.n_classes}")
    print(f"Number of events: {train_dataset.n_events}")
    
    # Example of accessing data
    sample = train_dataset[0]
    print(f"Sample data shape: {sample['data'].shape}")
    print(f"Sample: {sample}")