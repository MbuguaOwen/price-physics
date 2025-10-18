param(
  [string]$ROOT="data/images_jan_jul",
  [string]$SNAP="$PWD/data/images_jan_jul/train_snapshot.csv"
)

# 1) Build a snapshot from finished months (edit include list as needed)
python scripts/make_snapshot.py --root $ROOT --out_csv $SNAP --include "BTCUSDT_2025-01,BTCUSDT_2025-02"

# 2) Sanity train on snapshot
python scripts/train_sanity.py --snapshot $SNAP --batch 64 --epochs 1

