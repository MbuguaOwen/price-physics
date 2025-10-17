# scripts/train_sanity.py
# --- repo bootstrap (add project root to sys.path) ---
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # parent of /scripts
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ------------------------------------------------------
from src.datasets.shard_dataset import ShardDataset

import argparse, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

class TinyCNN(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Linear(32, 2)  # dummy 2-class head

    def forward(self, x):
        z = self.net(x)
        z = z.view(z.size(0), -1)
        return self.head(z)

def infer_channels(snapshot_csv):
    import pandas as pd, numpy as np
    df = pd.read_csv(snapshot_csv, nrows=1)
    shard = df["fullpath"].iloc[0]
    idx = int(df["idx_in_shard"].iloc[0])
    arr = np.load(shard, mmap_mode="r")
    return int(arr.shape[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="data/images_jan_jul/train_snapshot.csv")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    ds = ShardDataset(args.snapshot)
    C = infer_channels(args.snapshot)
    model = TinyCNN(C)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=0, persistent_workers=False)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
            if total > 5:  # just a short warm-up
                break
        print(f"epoch={epoch} warmup_loss={total:.4f}")

if __name__ == "__main__":
    main()
