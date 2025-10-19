# price-physics — Pure Price Physics Crypto Prediction System

**Status:** Production scaffold with full pipeline code and replication instructions.  
**Note:** The `outputs/` included here are **example placeholders** to demonstrate structure only.
To produce your **final OOS results and motif library**, follow the Runbook below with your market data.

--- 

## System Overview
This repository implements an end-to-end pipeline that:
1. Builds **information-time bars** (dollar bars; optional imbalance bars) from raw ticks.
2. Generates **image features** (GAF/GADF/MTF) from rolling price windows.
3. Applies **triple-barrier labeling** with volatility-scaled PT/SL and timeouts.
4. Trains a **ResNet-lite CNN** for 3-class (+1/0/-1) directional predictions.
5. Runs **anchored walk-forward** evaluation and converts probabilities to **PnL** after costs.
6. Produces an **interpretability motif library** (Grad-CAM tiles + human-readable descriptors).

The design follows the project charter and definitions of done you approved: information-time bars,
proper OOS walk-forward, calibration, and motif extraction for stable edge.

---

## Environment
- Python ≥ 3.10
- Recommended: create a fresh virtualenv

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Requirements
See `requirements.txt`. Heavy packages (PyTorch, pyts, scikit-learn) are included.
If you only want to test the scaffolding without heavy training, you can skip those installs.

---

## Data Expectations
- Raw REAL ticks in `data/ticks_raw/` (Jan–Jul only; no synthetic) with columns:
  - `timestamp` (ns or ms epoch or ISO8601), `price` (float), `qty` (float), optional `is_buyer_maker` (bool/int).
- Example glob: `data/ticks_raw/BTCUSDT_2025-*.csv`

> You can also adapt `src/utils/data.py` loaders if your vendor schema differs.

---

## Runbook (Reproduce End-to-End)
1) **Bars**
```bash
python scripts/make_bars.py   --ticks_glob "data/ticks_raw/*2025-0[1-7]*.csv"   --out_dir "data/bars_dollar"   --bar_type dollar   --dollar_value 500000
```

2) **Labels (Triple-Barrier)**
```bash
python scripts/make_labels.py   --bars_glob "data/bars_dollar/*.parquet"   --out_dir "data/labels"   --config "configs/tbm.yaml"
```

3) **Imaging (GAF/GADF/MTF)**
```bash
python scripts/make_images.py   --bars_glob "data/bars_dollar/*.parquet"   --out_dir "data/images"   --config "configs/imaging.yaml"
```

4) **Train Model**
```bash
python scripts/train.py   --config "configs/train.yaml"   --data_root "data"   --artifacts_dir "outputs/artifacts"
```

5) **Walk-Forward + PnL**
```bash
python scripts/wf_eval.py   --config "configs/wf.yaml"   --data_root "data"   --outputs_dir "outputs"
```

6) **Interpretability / Motif Library**
```bash
python scripts/interpret.py   --config "configs/train.yaml"   --model_path "outputs/artifacts/model_last_fold.pt"   --images_root "data/images"   --out_dir "outputs/motifs"
```

# from project root { MODULES }
```bash
python -m scripts.demo_generate_ticks --symbol BTCUSDT --days 2 --out data/ticks_raw/BTCUSDT_demo.csv

python -m scripts.make_bars --ticks_glob "data/ticks_raw/*2025-0[1-7]*.csv" --out_dir data/bars_dollar --bar_type dollar --dollar_value 100000

python -m scripts.make_labels --bars_glob "data/bars_dollar/*.parquet" --out_dir data/labels --config configs/tbm.yaml

python -m scripts.make_images --bars_glob "data/bars_dollar/*.parquet" --out_dir data/images --config configs/imaging.yaml

python -m scripts.train --config configs/train.yaml --data_root data --artifacts_dir outputs/artifacts

python -m scripts.wf_eval --config configs/wf.yaml --data_root data --outputs_dir outputs

python -m scripts.interpret --config configs/train.yaml --model_path outputs/artifacts/model_last_fold.pt --images_root data/images --out_dir outputs/motifs

---


## Final Deliverables (after running on your data)
- `outputs/artifacts/` — Final trained model(s) from last fold(s)
- `outputs/reports/` — Walk-forward OOS reports (JSON + Markdown), confusion matrices, metrics
- `outputs/plots/` — Equity curve, drawdown, turnover, calibration plots
- `outputs/motifs/` — Motif gallery tiles + descriptors (Phase 5 deliverable)

Everything is parameterized via YAML in `configs/` for reproducibility.  
Seed control and fold manifests are saved in `outputs/reports/`.

---

## Reproducibility
- Deterministic seeds where supported (PyTorch, numpy, Python).
- Versioned configs and fold manifests.
- Cost model and slippage baked into the PnL step (see `configs/wf.yaml`).

---

## Folder Layout
```
price-physics/
  configs/
  data/
  docs/
  outputs/
  scripts/
  src/
```
See inline docstrings for module-level details.

---

## License & Attribution
For internal research and production trading use. No warranty. Use at your own risk.

1) Make src importable (one-time per shell)
$env:PYTHONPATH = "$PWD"


(Do this before running any script that imports from src/...)


# from project root { MODULES }

```bash

2) Estimate per-month TP/SL policy
python scripts\quantile_tp_sl_grouped.py `
  --snapshot "data\images_jan_jul\btc_snapshot_2025_01_07_FIXED.csv" `
  --horizon_minutes 600 `
  --sample 50000 `
  --atr_window 14 `
  --group month `
  --q_tp 0.60 --q_sl 0.40 `
  --out_json "outputs/tp_sl_policy_by_month_q60_40.json"


(If you want, generate a couple variants to compare later:)

python scripts\quantile_tp_sl_grouped.py ... --q_tp 0.55 --q_sl 0.45 --out_json "outputs/tp_sl_policy_by_month_q55_45.json"
python scripts\quantile_tp_sl_grouped.py ... --q_tp 0.65 --q_sl 0.35 --out_json "outputs/tp_sl_policy_by_month_q65_35.json"

3) Relabel from ticks using that policy
python -u scripts\label_from_ticks.py `
  --snapshot "data\images_jan_jul\btc_snapshot_2025_01_07_FIXED.csv" `
  --out_csv "data\images_jan_jul\btc_labels_ticksONLY_GROUPED_m600.csv" `
  --ticks_root "data\ticks_raw" `
  --bars_root "data\bars_dollar" `
  --atr_window 14 `
  --tp_mult 50.50335478971694 `
  --sl_mult 18.404720361751057 `
  --horizon_minutes 600 `
  --mode ticks `
  --tp_sl_policy_json "outputs/tp_sl_policy_by_month_q60_40.json" `
  --policy_group month `
  --resume

4) Uniqueness weights
python scripts\compute_uniqueness.py `
  --snapshot_labeled "data\images_jan_jul\btc_labels_ticksONLY_GROUPED_m600.csv" `
  --out_csv "data\images_jan_jul\btc_labels_ticksONLY_GROUPED_m600_w.csv"

5) CV folds (purged + embargoed)
python scripts\make_folds.py `
  --snapshot_labeled "data\images_jan_jul\btc_labels_ticksONLY_GROUPED_m600_w.csv" `
  --out_csv "data\images_jan_jul\btc_cv_folds_ticksONLY_GROUPED_m600.csv" `
  --k 5 --purge_time "1h" --embargo_time "600m"

6) (Re)build tiny regime features (or reuse if already built)
python scripts\make_regime_features.py `
  --snapshot "data\images_jan_jul\btc_snapshot_2025_01_07_FIXED.csv" `
  --out_csv "data\images_jan_jul\btc_features_m60_v60.csv" `
  --bars_root "data\bars_dollar" `
  --horiz_min 60 --vol_min 60


Your trainer should already merge btc_features_m60_v60.csv by row_id after the code change.

7) Retrain with the binary TP head (+ focal)
$CW_BIN = '{"0":1.0,"1":4.0}'

python scripts\train_classifier_cv.py `
  --snapshot_labeled "data\images_jan_jul\btc_labels_ticksONLY_GROUPED_m600_w.csv" `
  --cv_folds "data\images_jan_jul\btc_cv_folds_ticksONLY_GROUPED_m600.csv" `
  --binary_head --n_classes 2 `
  --batch 128 --epochs 4 --steps_per_epoch 300 `
  --sample_weight_col uniqueness `
  --class_weight "$CW_BIN" `
  --use_focal --focal_gamma 2.0 `
  --preds_out_binary "outputs/preds_cv_tp_grouped.csv" `
  --progress

8) Evaluate lift & EV (binary)
@'
import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score

P = pd.read_csv(r"outputs/preds_cv_tp_grouped.csv")
y = P["y_true_bin"].to_numpy().astype(int)
p = P["p_tp"].clip(0,1).to_numpy()

base = y.mean()
ap = average_precision_score(y,p)
print({"base_rate": round(base,4), "PR_AUC": round(ap,4), "lift": round(ap/base,2)})

W, L = 50.50335478971694, 18.404720361751057
def ev(prec): return prec*W - (1-prec)*L

best=None
for t in np.linspace(0,1,201):
    sel = p>=t
    if sel.sum()==0: continue
    prec = y[sel].mean(); e = ev(prec)
    if best is None or e>best[2]: best=(t,prec,e,sel.sum())
print({"breakeven_precision": round(L/(W+L),4),
       "thr": round(best[0],3), "precision": round(best[1],3),
       "EV_per_trade": round(best[2],3), "n_signals": int(best[3])})
'@ | python -

Optional: month/regime summary of EV at a threshold

(Just swap the CSV path if you try different runs.)

@'
import pandas as pd, numpy as np
P = pd.read_csv(r"outputs/preds_cv_tp_grouped.csv")
L = pd.read_csv(r"data/images_jan_jul/btc_labels_ticksONLY_GROUPED_m600_w.csv")
L = L.reset_index(drop=True); L["row_id"]=L.index.astype("int64")
M = P.merge(L[["row_id","month"]], on="row_id", how="left")

W, L0 = 50.50335478971694, 18.404720361751057
def EV(pr): return pr*W - (1-pr)*L0

thr = 0.5  # or use the best threshold printed above
rows=[]
for m,g in M.groupby("month"):
    y = g["y_true_bin"].to_numpy().astype(int)
    p = g["p_tp"].clip(0,1).to_numpy()
    sel = p>=thr
    if sel.sum()==0: rows.append((m,0,0.0,0.0)); continue
    pr = y[sel].mean()
    rows.append((m, int(sel.sum()), float(pr), float(EV(pr))))
print(pd.DataFrame(rows, columns=["month","n_trades","precision","EV_per_trade"]).to_string(index=False))
'@ | python -