import argparse, os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from src.datasets.shard_dataset import ShardDataset


class SmallCNN(nn.Module):
    def __init__(self, C, n_classes=3):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Linear(128, n_classes)
    def forward(self, x):
        z = self.feat(x); z = z.view(z.size(0), -1)
        return self.head(z)


def infer_C(snapshot_csv):
    df = pd.read_csv(snapshot_csv, nrows=1)
    arr = np.load(df["fullpath"].iloc[0], mmap_mode="r")
    return int(arr.shape[1])


def build_indices(df_folds, fold_id):
    is_train = (df_folds["fold"] == fold_id) & (df_folds["split"] == "train")
    is_test  = (df_folds["fold"] == fold_id) & (df_folds["split"] == "test")
    return np.flatnonzero(is_train.values), np.flatnonzero(is_test.values)


def run_fold(args, fold_id):
    # Load labeled + weights
    ds = ShardDataset(args.snapshot_labeled, label_col="label")
    folds = pd.read_csv(args.cv_folds)
    tr_idx, te_idx = build_indices(folds, fold_id)
    if len(tr_idx) == 0 or len(te_idx) == 0:
        print(f"[fold {fold_id}] empty split; skipping")
        return None

    ds_tr = Subset(ds, tr_idx)
    ds_te = Subset(ds, te_idx)

    # Build weighted sampler from uniqueness
    # Extract weights from underlying dataset rows
    w_tr = torch.tensor([float(ds.df.iloc[i]["uniqueness"]) for i in tr_idx], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights=w_tr, num_samples=min(len(w_tr), args.steps_per_epoch*args.batch), replacement=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = infer_C(args.snapshot_labeled)
    model = SmallCNN(C, n_classes=args.n_classes).to(device)

    loader_tr = DataLoader(ds_tr, batch_size=args.batch, sampler=sampler, num_workers=0)
    loader_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=0)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train(); tr_loss = 0.0
        steps = 0
        for xb, yb, wb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward(); opt.step()
            tr_loss += float(loss.item()); steps += 1
            if steps >= args.steps_per_epoch: break

        # Eval
        model.eval(); te_loss = 0.0; correct = 0; total = 0
        with torch.no_grad():
            for xb, yb, wb in loader_te:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb); loss = loss_fn(logits, yb)
                te_loss += float(loss.item())
                pred = logits.argmax(1); correct += int((pred==yb).sum()); total += yb.numel()

        print(f"[fold {fold_id}] epoch={ep} train_loss={tr_loss:.4f} test_loss={te_loss:.4f} test_acc={correct/max(1,total):.3f}")

    return {"fold": fold_id, "test_acc": correct/max(1,total), "test_loss": te_loss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot_labeled", required=True)
    ap.add_argument("--cv_folds", required=True)
    ap.add_argument("--n_classes", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps_per_epoch", type=int, default=200)  # cap per epoch for speed
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fold", type=int, default=-1)  # -1 means run all folds and average
    args = ap.parse_args()

    results = []
    if args.fold >= 0:
        r = run_fold(args, args.fold)
        if r: results.append(r)
    else:
        folds = pd.read_csv(args.cv_folds)["fold"].unique().tolist()
        for f in sorted(folds):
            r = run_fold(args, f)
            if r: results.append(r)

    if results:
        df = pd.DataFrame(results)
        print("CV summary:\n", df.describe(include="all"))

if __name__ == "__main__":
    main()

