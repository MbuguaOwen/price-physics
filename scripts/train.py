import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, os, glob, yaml, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from src.models.resnet_lite import build_model as build_resnet_lite
from src.models.timm_wrapper import build_timm
from tqdm.auto import tqdm
from src.utils.seeds import set_global_determinism


def compute_loss(logits, y, w=None):
    import torch.nn.functional as F
    loss = F.cross_entropy(logits, y, reduction='none')
    if w is not None:
        w = torch.clamp(w, 0.25, 4.0)
        loss = loss * w
    return loss.mean()

class ImageLabelDS(Dataset):
    def __init__(self, images_root, labels_root):
        imgs = sorted(glob.glob(os.path.join(images_root, "*_images.npy")))
        labs = sorted(glob.glob(os.path.join(labels_root, "*_labels.parquet")))
        import pandas as pd
        Xs, ys, ws = [], [], []
        for imgf in imgs:
            base = os.path.basename(imgf).replace("_images.npy","")
            labf = None
            for l in labs:
                if os.path.basename(l).startswith(base.split("_w")[0]):
                    labf = l; break
            if labf is None: continue
            X = np.load(imgf)  # (N, C, H, W)
            lab = pd.read_parquet(labf)
            m = min(len(X), len(lab))
            Xs.append(torch.tensor(X[:m]))
            ys.append(torch.tensor(lab["label"].values[:m], dtype=torch.long))
            if "sample_w" in lab.columns:
                ws.append(torch.tensor(lab["sample_w"].values[:m], dtype=torch.float))
        self.X = torch.cat(Xs, dim=0) if Xs else torch.zeros(0,4,64,64)
        self.y = torch.cat(ys, dim=0) if ys else torch.zeros(0, dtype=torch.long)
        self.w = torch.cat(ws, dim=0) if ws else None
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        if self.w is not None:
            return self.X[i], self.y[i], self.w[i]
        return self.X[i], self.y[i]

def infer_in_ch(images_root):
    import numpy as np, glob, os
    files = glob.glob(os.path.join(images_root, "*_images.npy"))
    if not files: return 4
    X = np.load(files[0])
    return int(X.shape[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--images_dir", default=None)
    ap.add_argument("--labels_dir", default=None)
    ap.add_argument("--val_labels_dir", default=None)
    ap.add_argument("--artifacts_dir", default="outputs/artifacts")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.artifacts_dir, exist_ok=True)

    train_images_dir = args.images_dir or os.path.join(args.data_root, "images")
    train_labels_dir = args.labels_dir or os.path.join(args.data_root, "labels")
    ds = ImageLabelDS(train_images_dir, train_labels_dir)
    if len(ds) == 0:
        tqdm.write("No images/labels. Generate data first.")
        torch.save({}, os.path.join(args.artifacts_dir, "model_last_fold.pt")); 
        return
    dl = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True)

    in_ch = int(cfg.get("in_channels",0)) or infer_in_ch(train_images_dir)
    if cfg.get("model_name","resnet_lite") == "resnet_lite":
        model = build_resnet_lite(num_classes=3, in_ch=in_ch)
    elif cfg["model_name"] in ["timm_convnext_tiny","timm_vit_tiny"]:
        model = build_timm(cfg["model_name"], num_classes=3, in_ch=in_ch)
    else:
        raise ValueError("unknown model")

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))

    best_loss = 1e9; patience = int(cfg.get("early_stop_patience",4)); wait = 0
    epochs = int(cfg["epochs"])
    for epoch in range(epochs):
        model.train(); running=0.0; n=0
        batch_iter = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False)
        for batch in batch_iter:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                xb, yb, wb = batch
            else:
                xb, yb = batch[0], batch[1]
                wb = None
            opt.zero_grad(); logits = model(xb); loss = compute_loss(logits, yb, wb); loss.backward(); opt.step()
            running += loss.item()*len(yb); n += len(yb)
            batch_iter.set_postfix({"loss": running/max(1,n)})
        epoch_loss = running/max(1,n)
        tqdm.write(f"Epoch {epoch+1}/{epochs} loss {epoch_loss:.4f}")
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss; wait = 0
            torch.save(model.state_dict(), os.path.join(args.artifacts_dir, "model_last_fold.pt"))
        else:
            wait += 1
            if wait >= patience: break
    tqdm.write(f"Saved best model to {os.path.join(args.artifacts_dir, 'model_last_fold.pt')}")

set_global_determinism(1337)

if __name__ == "__main__":
    main()
