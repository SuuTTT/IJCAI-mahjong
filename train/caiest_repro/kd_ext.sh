#!/bin/bash
# Extend kd0 + kdens gates from 6 to 24 blocks to resolve kd0's positive mean.
cd /root/IJCAI-mahjong/train/caiest_repro
A=ckpt/aug/aug_128x40_s0.pkl
K0=ckpt/kd/kd_128x40_s0.pkl; K1=ckpt/kd/kd_128x40_s1.pkl; K2=ckpt/kd/kd_128x40_s2.pkl
for i in $(seq 6 23); do
  while [ "$(awk '{print ($1>105)?1:0}' /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f kd_blocks/kd0_b$i.json ]   || python3 e8_gate.py --cand $K0 --ref $A --lam 0 --seeds 500 --workers 60 --seed0 $((340000 + i*1000)) --out kd_blocks/kd0_b$i.json
  [ -f kd_blocks/kdens_b$i.json ] || python3 e12_ens_gate.py --cand $K0,$K1,$K2 --ref $A --seeds 500 --workers 60 --seed0 $((350000 + i*1000)) --out kd_blocks/kdens_b$i.json
done
python3 - << "PYEOF"
import json, glob, sys, numpy as np
T = {5: 2.571, 23: 2.069}
out = {}; fail = []
for tag, want in [("kd0", 24), ("kdens", 24), ("kd1", 6), ("kd2", 6)]:
    fs = sorted(glob.glob(f"kd_blocks/{tag}_b*.json"))
    if len(fs) != want:
        fail.append(f"{tag}: expected {want} blocks, found {len(fs)}")
        continue
    v = np.array([json.load(open(f))["placement_pts"] for f in fs])
    n = len(v)
    se = v.std(ddof=1) / np.sqrt(n)
    lo = v.mean() - T.get(n - 1, 2.069) * se
    out[tag] = dict(n_blocks=n, mean=round(float(v.mean()), 4),
                    ci95_lo=round(float(lo), 4), margin_lo=round(float(lo - 2.5), 4),
                    blocks=[round(float(x), 4) for x in v],
                    verdict="BEATS_AUGS0" if lo > 2.5 else "TIED_NOT_SEPARATED")
out["integrity"] = "OK" if not fail else "LOUDFAIL: " + "; ".join(fail)
json.dump(out, open("KD_EXT_RESULTS.json", "w"), indent=2)
print(json.dumps(out, indent=2))
if fail: sys.exit(1)
PYEOF
echo "KD EXT DONE"
