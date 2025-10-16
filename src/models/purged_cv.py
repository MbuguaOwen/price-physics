from __future__ import annotations
import numpy as np, pandas as pd

class PurgedKFoldTime:
    """
    Time-ordered K-Fold that PURGES training samples overlapping test intervals
    and applies an EMBARGO after each test fold.
    Args:
      n_splits: number of folds
      embargo: pd.Timedelta, e.g., pd.Timedelta(minutes=5)
      t0: pd.Series of start times (UTC); index aligns to samples
      t1: pd.Series of end times (UTC); index aligns to samples
    """
    def __init__(self, n_splits: int = 5, embargo: "pd.Timedelta | None" = None):
        self.n_splits = int(n_splits)
        self.embargo = embargo

    def split(self, X, t0: pd.Series, t1: pd.Series):
        idx = np.asarray(getattr(X, "index", np.arange(len(t0))), dtype=object)
        order = np.argsort(t0.values)  # chronological
        idx_sorted = idx[order]
        t0s, t1s = t0.values[order], t1.values[order]
        n = len(idx_sorted)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        starts = np.cumsum(np.concatenate(([0], fold_sizes[:-1])))
        for k, (start, size) in enumerate(zip(starts, fold_sizes)):
            test_mask = np.zeros(n, dtype=bool)
            test_mask[start:start+size] = True
            test_idx = idx_sorted[test_mask]

            # Determine test interval [t0_min, t1_max]
            t0_min = t0s[test_mask].min()
            t1_max = t1s[test_mask].max()

            # Purge: drop any train with (t0 < t1_test_max) & (t1 > t0_test_min)
            train_mask = ~test_mask
            overlap = (t0s < t1_max) & (t1s > t0_min)
            train_mask &= ~overlap

            # Embargo: drop train samples that start within embargo after t1_max
            if self.embargo is not None:
                embargo_end = t1_max + self.embargo
                train_mask &= ~( (t0s >= t1_max) & (t0s < embargo_end) )

            yield idx_sorted[train_mask], test_idx

