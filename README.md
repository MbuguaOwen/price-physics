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
- Raw ticks in `data/ticks_raw/` with columns:
  - `timestamp` (ns or ms epoch or ISO8601), `price` (float), `qty` (float), optional `is_buyer_maker` (bool/int).
- Example glob: `data/ticks_raw/BTCUSDT_2025-*.csv`

> You can also adapt `src/utils/data.py` loaders if your vendor schema differs.

---

## Runbook (Reproduce End-to-End)
1) **Bars**
```bash
python scripts/make_bars.py   --ticks_glob "data/ticks_raw/*.csv"   --out_dir "data/bars_dollar"   --bar_type dollar   --dollar_value 500000
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

python -m scripts.make_bars --ticks_glob "data/ticks_raw/*.csv" --out_dir data/bars_dollar --bar_type dollar --dollar_value 100000

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
