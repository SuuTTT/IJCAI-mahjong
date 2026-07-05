#!/bin/bash
# CONFIRMATION round for kdens (3-KD mean-softmax ensemble) on FRESH seeds.
# 24 independent blocks, seed range disjoint from all prior gates (360000+).
cd /root/IJCAI-mahjong/train/caiest_repro
A=ckpt/aug/aug_128x40_s0.pkl
K0=ckpt/kd/kd_128x40_s0.pkl; K1=ckpt/kd/kd_128x40_s1.pkl; K2=ckpt/kd/kd_128x40_s2.pkl
mkdir -p kd_blocks
for i in $(seq 0 23); do
  while [ "$(awk '{print ($1>105)?1:0}' /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f kd_blocks/kdconf_b$i.json ] || python3 e12_ens_gate.py --cand $K0,$K1,$K2 --ref $A --seeds 500 --workers 60 --seed0 $((360000 + i*1000)) --out kd_blocks/kdconf_b$i.json
done
python3 - << "PYEOF"
import json, glob, sys, numpy as np
fs = sorted(glob.glob("kd_blocks/kdconf_b*.json"))
out = {}
if len(fs) != 24:
    out["integrity"] = f"LOUDFAIL: expected 24 blocks, found {len(fs)}"
else:
    v = np.array([json.load(open(f))["placement_pts"] for f in fs])
    n = len(v); se = v.std(ddof=1)/np.sqrt(n); lo = v.mean() - 2.069*se
    out = dict(candidate="kdens 3xKD mean-softmax", n_blocks=n,
               mean=round(float(v.mean()),4), ci95_lo=round(float(lo),4),
               margin_lo=round(float(lo-2.5),4), blocks=[round(float(x),4) for x in v],
               verdict="CONFIRMED_BEATS_AUGS0" if lo>2.5 else "NOT_CONFIRMED",
               integrity="OK")
json.dump(out, open("KDENS_CONFIRM.json","w"), indent=2)
print(json.dumps(out, indent=2))
if out.get("integrity") != "OK": sys.exit(1)
PYEOF
echo "KDENS CONFIRM DONE"
