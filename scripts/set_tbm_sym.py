import yaml, sys

p = "configs/tbm.yaml"
c = yaml.safe_load(open(p, encoding="utf-8")) or {}

# Example symmetric setting; adjust via CLI args
c["barriers"] = (c.get("barriers", {}) | {"mode": "atr"})
c["vol_method"] = "atr"
c["pt_mult"] = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
c["sl_mult"] = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
mins = int(sys.argv[3]) if len(sys.argv) > 3 else 20
c["horizon"] = {"type": "clock", "minutes": mins}

open(p, "w", encoding="utf-8").write(yaml.safe_dump(c, sort_keys=False))
print(f"tbm.yaml set: pt_mult={c['pt_mult']} sl_mult={c['sl_mult']} horizon={c['horizon']}")

