import argparse, os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, json
import ast
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from src.datasets.shard_dataset import ShardDataset
from src.utils.progress import pbar


def parse_class_weight(s, n_classes=None):
    if not s:
        return None
    try:
        d = json.loads(s)
        return {int(k): float(v) for k, v in d.items()}
    except Exception:
        pass
    try:
        d = ast.literal_eval(s)
        return {int(k): float(v) for k, v in d.items()}
    except Exception:
        pass
    try:
        d = {}
        s2 = s.strip().strip("{}")
        if s2:
            for p in s2.split(","):
                k, v = p.split(":")
                k = k.strip().strip('"').strip("'")
                v = v.strip()
                d[int(k)] = float(v)
        if n_classes is not None and not all(i in d for i in range(n_classes)):
            print(f"[warn] class_weight missing some classes 0..{n_classes-1}: got {sorted(d.keys())}")
        return d if d else None
    except Exception as e:
        print(f"[warn] failed to parse --class_weight={s}: {e}")
        return None


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


def build_split_for_fold(L: pd.DataFrame, F: pd.DataFrame, f: int):
    mask = F["fold"] == f
    Ff = F.loc[mask, ["row_id","split"]].copy()

    tr_ids = Ff.loc[Ff["split"]=="train", "row_id"].to_frame()
    val_ids = Ff.loc[Ff["split"]=="val", "row_id"].to_frame()

    train_df = tr_ids.merge(L, on="row_id", how="inner", copy=False)
    val_df   = val_ids.merge(L, on="row_id", how="inner", copy=False)

    if len(train_df)==0 or len(val_df)==0:
        raise SystemExit(f"Fold {f}: empty train/val after merge. Check folds and snapshot alignment.")

    return train_df, val_df


def run_fold(args, fold_id, L: pd.DataFrame, F: pd.DataFrame):
    # Build keyed train/val splits
    train_df, val_df = build_split_for_fold(L, F, fold_id)

    # Dataset built from snapshot; map row_id -> positions in ds.df
    ds = ShardDataset(args.snapshot_labeled, label_col="label")
    # Ensure ds.df carries row_id by joining on stable keys
    if "row_id" not in ds.df.columns:
        join_cols = [c for c in ("fullpath","idx_in_shard") if c in ds.df.columns and c in L.columns]
        if join_cols:
            ds.df = ds.df.merge(L[join_cols + ["row_id"]], on=join_cols, how="left")
        else:
            # Fallback: assume order alignment (best-effort)
            ds.df = ds.df.reset_index(drop=True)
            ds.df.insert(0, "row_id", ds.df.index.astype("int64"))

    # Optional: join tiny "regime" features by row_id if available
    try:
        features_csv = "data/images_jan_jul/btc_features_m60_v60.csv"
        if os.path.exists(features_csv):
            Fextra = pd.read_csv(features_csv)
            Fextra["row_id"] = Fextra["row_id"].astype("int64")
            ds.df = ds.df.reset_index(drop=True)
            if "row_id" not in ds.df.columns:
                ds.df["row_id"] = ds.df.index.astype("int64")
            ds.df = ds.df.merge(Fextra, on="row_id", how="left")
            for col in ("ret_m","vol_m","hour"):
                if col in ds.df.columns:
                    ds.df[col] = ds.df[col].fillna(0.0)
            if all(c in ds.df.columns for c in ("ret_m","vol_m","hour")):
                desc = ds.df[["ret_m","vol_m","hour"]].describe().to_dict()
                print("[info] joined regime features:", desc)
    except Exception as e:
        print(f"[warn] failed to join regime features: {e}")

    # Configure binary-head mode: derive y_bin from 3-class label
    if getattr(args, "binary_head", False):
        try:
            ds.df["y_bin"] = (ds.df["label"].astype(int) == 1).astype("int64")
            ds.label_col = "y_bin"
            n_classes_eff = 2
        except Exception as e:
            raise SystemExit(f"--binary_head requested but failed to derive y_bin: {e}")
    else:
        n_classes_eff = args.n_classes

    # Map row_id -> position in ds.df, filter unknown ids
    pos_by_rid = {int(r): i for i, r in enumerate(ds.df["row_id"].astype("int64").tolist())}
    tr_pos = np.array([pos_by_rid.get(int(r), -1) for r in train_df["row_id"].tolist()], dtype=np.int64)
    val_pos = np.array([pos_by_rid.get(int(r), -1) for r in val_df["row_id"].tolist()], dtype=np.int64)
    tr_pos = tr_pos[tr_pos >= 0]
    val_pos = val_pos[val_pos >= 0]
    if len(tr_pos)==0 or len(val_pos)==0:
        print(f"[fold {fold_id}] empty split after mapping to dataset; skipping")
        return None

    ds_tr = Subset(ds, tr_pos)
    ds_te = Subset(ds, val_pos)

    # ---------- weights setup ----------
    # Optional class weights (string -> dict[int,float])
    class_weight_tensor = None
    class_weight_map = parse_class_weight(getattr(args, "class_weight", None), n_classes=n_classes_eff)
    if class_weight_map:
        class_weight_tensor = torch.tensor([class_weight_map.get(i, 1.0) for i in range(n_classes_eff)], dtype=torch.float32)
    # -----------------------------------

    # Build weighted sampler using per-sample weights if available; fallback to uniqueness/ones
    sample_weight_col = getattr(args, "sample_weight_col", None)
    def weight_for_pos(i):
        if sample_weight_col and sample_weight_col in ds.df.columns:
            try:
                return float(ds.df.iloc[i][sample_weight_col])
            except Exception:
                return 1.0
        return float(ds.df.iloc[i]["uniqueness"]) if "uniqueness" in ds.df.columns else 1.0

    # class balancing + de-overlap sampler: inverse class frequency times uniqueness
    # Build weights per row_id from train_df, then align to subset order
    cls_counts = train_df["label"].value_counts().to_dict()
    cls_inv = train_df["label"].map(lambda c: 1.0/cls_counts.get(c, 1.0)).astype(float)
    uni = train_df["uniqueness"].astype(float) if "uniqueness" in train_df.columns else pd.Series(1.0, index=train_df.index)
    samp_weights_df = (cls_inv * uni + 1e-12)
    rid_to_w = dict(zip(train_df["row_id"].astype(int).to_numpy(), samp_weights_df.to_numpy()))
    # Align in ds_tr order using row_ids we extracted
    tr_row_ids = ds.df.loc[tr_pos, "row_id"].astype("int64").to_numpy()
    w_tr = torch.tensor([rid_to_w.get(int(r), 1.0) for r in tr_row_ids], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights=w_tr, num_samples=min(len(w_tr), args.steps_per_epoch*args.batch), replacement=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = infer_C(args.snapshot_labeled)
    model = SmallCNN(C, n_classes=n_classes_eff).to(device)

    # Ensure we can recover row_ids from the dataset for sample weights
    class WithRowId(torch.utils.data.Dataset):
        def __init__(self, base, row_ids: np.ndarray):
            self.base = base
            self.row_ids = row_ids
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            out = self.base[i]
            x, y = out[0], out[1]
            return x, y, int(self.row_ids[i])

    # Row IDs aligned to subset order
    tr_row_ids = ds.df.loc[tr_pos, "row_id"].astype("int64").to_numpy()
    te_row_ids = ds.df.loc[val_pos, "row_id"].astype("int64").to_numpy()

    ds_tr = WithRowId(ds_tr, tr_row_ids)
    ds_te = WithRowId(ds_te, te_row_ids)

    loader_tr = DataLoader(ds_tr, batch_size=args.batch, sampler=sampler, num_workers=0, drop_last=False)
    loader_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=0, drop_last=False)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Weighted loss configuration
    criterion = nn.CrossEntropyLoss(reduction="none")
    if class_weight_tensor is not None:
        class_weight_tensor = class_weight_tensor.to(device)

    def focal_ce_loss(logits, y, sample_w=None, class_weight_tensor=None, gamma=2.0):
        """
        Multiclass focal cross-entropy with optional class weights and per-sample weights.
        logits: [B, C], y: [B] Long
        """
        logp = torch.log_softmax(logits, dim=1)
        p = torch.softmax(logits, dim=1)
        idx = torch.arange(y.shape[0], device=logits.device)
        logp_t = logp[idx, y]
        p_t = p[idx, y]
        ce = -logp_t
        focal = (1.0 - p_t).pow(float(getattr(args, "focal_gamma", 2.0)))
        loss = focal * ce
        if class_weight_tensor is not None:
            loss = loss * class_weight_tensor[y]
        if sample_w is not None:
            loss = loss * sample_w
        return loss.mean()

    def loss_fn(logits, y, sample_w=None):
        if getattr(args, "use_focal", False):
            return focal_ce_loss(logits, y, sample_w, class_weight_tensor, gamma=getattr(args, "focal_gamma", 2.0))
        ce = criterion(logits, y)  # [B]
        if class_weight_tensor is not None:
            ce = ce * class_weight_tensor[y]
        if sample_w is not None:
            ce = ce * sample_w
        return ce.mean()

    # Build row_id -> sample weight maps (train/val)
    sw_map_tr = None
    sw_map_te = None
    if sample_weight_col and sample_weight_col in train_df.columns:
        sw_map_tr = dict(zip(train_df["row_id"].astype(int).to_numpy(), train_df[sample_weight_col].astype(float).to_numpy()))
    if sample_weight_col and sample_weight_col in val_df.columns:
        sw_map_te = dict(zip(val_df["row_id"].astype(int).to_numpy(), val_df[sample_weight_col].astype(float).to_numpy()))

    bar_e = pbar(total=args.epochs, desc=f"fold {fold_id} epochs", position=1, leave=True) if getattr(args, "progress", False) else None
    print(f"[info] sample_weight_col={getattr(args, 'sample_weight_col', None)}")
    print(f"[info] class_weight={getattr(args, 'class_weight', None)}")

    rows_bin = []  # collect binary-head predictions for this fold (only last epoch)

    for ep in range(args.epochs):
        model.train(); tr_loss = 0.0
        steps = 0
        for xb, yb, row_ids in loader_tr:
            xb, yb = xb.to(device), yb.to(device)

            batch_w = None
            if sw_map_tr is not None:
                rids = row_ids.detach().cpu().numpy() if hasattr(row_ids, "detach") else np.asarray(row_ids)
                batch_w = torch.tensor([sw_map_tr.get(int(r), 1.0) for r in rids], dtype=torch.float32, device=device)
                # normalize to keep loss scale reasonable
                nz = batch_w > 0
                if torch.any(nz):
                    batch_w = batch_w / (batch_w[nz].mean() + 1e-12)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)

            # One-time debug on first batch of fold 0
            if fold_id == 0 and 'did_debug' not in globals():
                try:
                    print("[debug] y unique in first batch:", torch.unique(yb).tolist())
                    sm = torch.softmax(logits, dim=1).mean(0).detach().cpu().numpy()
                    print("[debug] mean softmax over batch:", {i: float(sm[i]) for i in range(sm.shape[0])})
                except Exception:
                    pass
                globals()['did_debug'] = True

            loss = loss_fn(logits, yb, batch_w)
            loss.backward(); opt.step()
            tr_loss += float(loss.item()); steps += 1
            if steps >= args.steps_per_epoch: break

        # Eval
        model.eval(); val_losses = []; correct = 0; total = 0

        # collectors for validation predictions
        val_row_ids_all = []
        val_y_all = []
        val_p_all = []

        with torch.no_grad():
            for xb, yb, row_ids in loader_te:
                xb, yb = xb.to(device), yb.to(device)
                batch_w = None
                if sw_map_te is not None:
                    rids = row_ids.detach().cpu().numpy() if hasattr(row_ids, "detach") else np.asarray(row_ids)
                    batch_w = torch.tensor([sw_map_te.get(int(r), 1.0) for r in rids], dtype=torch.float32, device=device)
                    nz = batch_w > 0
                    if torch.any(nz):
                        batch_w = batch_w / (batch_w[nz].mean() + 1e-12)
                logits = model(xb)
                loss = loss_fn(logits, yb, batch_w)
                val_losses.append(loss.item())
                pred = logits.argmax(1); correct += int((pred==yb).sum()); total += yb.numel()

                # collect predictions
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                val_p_all.append(probs)
                val_row_ids_all.append((row_ids.detach().cpu().numpy() if hasattr(row_ids, "detach") else np.asarray(row_ids)))
                val_y_all.append(yb.detach().cpu().numpy())

                # Optional: collect binary-head preds (TP vs rest) on last epoch only
                if getattr(args, "binary_head", False) and ep == (args.epochs - 1):
                    try:
                        row_id_cpu = (row_ids.detach().cpu().numpy() if hasattr(row_ids, "detach") else np.asarray(row_ids)).astype("int64")
                        p_tp = probs[:, 1]
                        y_true_bin = yb.detach().cpu().numpy().astype("int64")
                        if getattr(args, "preds_out_binary", None):
                            rows_bin.append(np.column_stack([
                                row_id_cpu, np.full_like(row_id_cpu, int(fold_id)), y_true_bin, p_tp
                            ]))
                    except Exception as e:
                        print(f"[warn] failed collecting binary preds: {e}")

        te_loss = float(np.mean(val_losses)) if val_losses else 0.0

        # ----- write per-fold validation preds -----
        if val_p_all:
            val_p_all = np.concatenate(val_p_all, axis=0)
            val_row_ids_all = np.concatenate(val_row_ids_all, axis=0)
            val_y_all = np.concatenate(val_y_all, axis=0)

            preds_df = pd.DataFrame({
                "row_id": val_row_ids_all.astype(np.int64),
                "fold": int(fold_id),
                "y_true": val_y_all.astype(np.int64),
            })
            for k in range(n_classes_eff):
                preds_df[f"p{k}"] = val_p_all[:, k]

            out_dir = os.path.dirname(args.preds_out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            mode = "a" if os.path.exists(args.preds_out) else "w"
            header = not os.path.exists(args.preds_out)
            preds_df.to_csv(args.preds_out, index=False, mode=mode, header=header)
            print(f"[fold {fold_id}] wrote {len(preds_df)} preds -> {args.preds_out}")
        # -------------------------------------------
        print(f"[fold {fold_id}] epoch={ep} train_loss={tr_loss:.4f} val_loss={te_loss:.4f} val_acc={correct/max(1,total):.3f}")
        if bar_e:
            bar_e.update(1)
    if bar_e:
        bar_e.close()

    # End of fold: write binary-head preds if requested
    if getattr(args, "binary_head", False) and getattr(args, "preds_out_binary", None) and len(rows_bin):
        try:
            import pandas as pd, numpy as np, os
            Z = np.vstack(rows_bin)
            dfp = pd.DataFrame(Z, columns=["row_id","fold","y_true_bin","p_tp"]).astype({
                "row_id": "int64", "fold": "int64", "y_true_bin": "int64", "p_tp": "float32"
            })
            out_dir = os.path.dirname(args.preds_out_binary)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            mode = "a" if os.path.exists(args.preds_out_binary) else "w"
            header = not os.path.exists(args.preds_out_binary)
            dfp.to_csv(args.preds_out_binary, index=False, mode=mode, header=header)
            print(f"[fold {fold_id}] wrote {len(dfp)} binary preds -> {args.preds_out_binary}")
        except Exception as e:
            print(f"[warn] failed to write binary preds: {e}")
    return {"fold": fold_id, "val_acc": correct/max(1,total), "val_loss": te_loss}


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
    ap.add_argument("--progress", action="store_true", help="Show tqdm progress bars")
    ap.add_argument("--sample_weight_col", type=str, default=None,
                    help="Optional: column name in snapshot_labeled with per-sample weights (e.g., 'uniqueness').")
    ap.add_argument("--class_weight", type=str, default=None,
                    help="Optional: JSON-like dict of class weights, e.g. '{0:0.69,1:1.86,2:0.99}'.")
    ap.add_argument("--preds_out", type=str, default="outputs/preds_cv.csv",
                    help="CSV path to append per-fold validation predictions (row_id, fold, y_true, p0..p2).")
    ap.add_argument("--use_focal", action="store_true",
                    help="Use focal loss instead of standard CE.")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--binary_head", action="store_true",
                    help="Use a TP-vs-rest binary head for training/metrics.")
    ap.add_argument("--preds_out_binary", type=str, default=None,
                    help="Write CV preds for the binary head here (row_id, fold, y_true_bin, p_tp).")
    args = ap.parse_args()

    # Load labeled dataset with stable row_id
    L = pd.read_csv(args.snapshot_labeled)
    L = L.reset_index(drop=True)
    if "row_id" not in L.columns:
        L.insert(0, "row_id", L.index.astype("int64"))

    # Load folds (long-form with row_id, fold, split)
    F = pd.read_csv(args.cv_folds)
    # Backward-compat: allow 'test' split to be treated as 'val'
    if "split" in F.columns:
        try:
            F["split"] = F["split"].replace({"test": "val"})
        except Exception:
            pass
    required_cols = {"row_id","fold","split"}
    missing = required_cols - set(F.columns)
    if missing:
        raise SystemExit(f"folds file missing columns: {missing}")

    # Optional: parse class/sample weights if present
    cw_info = parse_class_weight(getattr(args, "class_weight", None), n_classes=args.n_classes)
    print(f"[info] class_weight parsed: {cw_info}")
    if getattr(args, "sample_weight_col", None) and args.sample_weight_col not in L.columns:
        print(f"[warn] --sample_weight_col '{args.sample_weight_col}' not found in labeled CSV; ignoring.")
        args.sample_weight_col = None

    print(f"[info] snapshot rows: {len(L)}; folds rows: {len(F)}; unique row_id in snapshot: {L['row_id'].nunique()}")
    print(f"[info] folds unique row_id: {F['row_id'].nunique()}; folds folds: {sorted(F['fold'].unique())}")

    results = []
    if args.fold >= 0:
        r = run_fold(args, args.fold, L, F)
        if r: results.append(r)
    else:
        folds = sorted(F["fold"].unique().tolist())
        bar = pbar(total=len(folds), desc="CV folds") if args.progress else None
        for f in folds:
            r = run_fold(args, f, L, F)
            if r: results.append(r)
            if bar:
                try:
                    bar.set_postfix_str(f"fold={f}")
                except Exception:
                    pass
                bar.update(1)
        if bar:
            bar.close()

    if results:
        df = pd.DataFrame(results)
        print("CV summary:\n", df.describe(include="all"))

if __name__ == "__main__":
    main()
