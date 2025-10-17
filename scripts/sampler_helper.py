from __future__ import annotations
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def make_sampler_from_labels(labels_parquet: str):
    df = pd.read_parquet(labels_parquet)
    if "sample_w" not in df.columns:
        raise FileNotFoundError("sample_w not found in labels; rerun materialize_folds or make_fold_weights.")
    weights = torch.as_tensor(df["sample_w"].values, dtype=torch.float)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

