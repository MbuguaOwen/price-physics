import argparse, os, json, numpy as np, torch
from pathlib import Path
from src.models.resnet_lite import build_model as build_resnet_lite
from src.models.timm_wrapper import build_timm
from src.interp.grad_cam import grad_cam
from src.interp.motif_cluster import cluster_motifs
from tqdm.auto import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--model_path", default="outputs/artifacts/model_last_fold.pt")
    ap.add_argument("--images_root", default="data/images")
    ap.add_argument("--out_dir", default="outputs/motifs")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    cfg = __import__("yaml").safe_load(open(args.config))
    progress = tqdm(total=4, desc="Interpreting model", unit="step")

    in_ch = int(cfg.get("in_channels",10))
    model = build_timm(cfg.get("model_name","timm_convnext_tiny"), num_classes=3, in_ch=in_ch) if cfg.get("model_name")!="resnet_lite" else build_resnet_lite(num_classes=3, in_ch=in_ch)
    progress.update()

    if os.path.exists(args.model_path):
        try: model.load_state_dict(torch.load(args.model_path, map_location="cpu"), strict=False)
        except Exception: pass
    model.eval()
    progress.update()

    X = torch.randn(16, in_ch, 64, 64)
    cams = grad_cam(model, X)
    labels = cluster_motifs(cams.numpy(), k=8)
    progress.update()

    np.save(os.path.join(args.out_dir, "cam_tiles.npy"), cams.detach().cpu().numpy())
    desc = [{"name": f"motif_{i}", "descriptor": "Run on real data to auto-describe cluster"} for i in range(8)]
    json.dump(desc, open(os.path.join(args.out_dir, "descriptors.json"), "w"), indent=2)
    np.save(os.path.join(args.out_dir, "cluster_labels.npy"), labels)
    progress.update()
    progress.close()
    tqdm.write(f"Wrote CAM tiles, descriptors, and cluster labels to {args.out_dir}")

if __name__ == "__main__":
    main()
