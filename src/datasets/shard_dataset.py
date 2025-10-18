# src/datasets/shard_dataset.py
import os, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

class ShardDataset(Dataset):
    def __init__(self, snapshot_csv: str, transform=None, label_col: str | None = None):
        df = pd.read_csv(snapshot_csv, low_memory=False)
        if "fullpath" not in df.columns and "relpath" in df.columns:
            root = os.path.dirname(snapshot_csv)
            df["fullpath"] = df["relpath"].apply(lambda p: os.path.join(root, p))
        if "fullpath" not in df.columns:
            raise ValueError("snapshot_csv must have 'fullpath' or 'relpath'.")
        df = df[df["fullpath"].apply(os.path.exists)].reset_index(drop=True)
        self.df = df
        self.transform = transform
        self.label_col = label_col
        self._cache = {}  # shard path -> memmap

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        shard = row["fullpath"]; idx = int(row["idx_in_shard"])
        arr = self._cache.get(shard)
        if arr is None:
            arr = np.load(shard, mmap_mode="r")  # (B,C,H,W) float32
            self._cache[shard] = arr
        x = np.array(arr[idx], dtype=np.float32)  # copy slice
        if self.transform: x = self.transform(x)
        x_t = torch.from_numpy(x)
        y = int(row[self.label_col]) if (self.label_col and self.label_col in row) else 0
        w = float(row["uniqueness"]) if "uniqueness" in row else 1.0
        return x_t, y, w
