#!/bin/bash
# Diversity-ensemble gates (prespecified, no cherry-picking):
#   kddiv4    = 4 diverse students alone
#   kdens7div = kdens3 originals + 4 diverse (7-model)
# Both 24 blocks at seed0 360000+ (paired vs kdconf/kdens6 ranges).
cd /root/IJCAI-mahjong/train/caiest_repro
A=ckpt/aug/aug_128x40_s0.pkl
K3=ckpt/kd/kd_128x40_s0.pkl,ckpt/kd/kd_128x40_s1.pkl,ckpt/kd/kd_128x40_s2.pkl
D4=ckpt/kddiv/kdd_heavyaug.pkl,ckpt/kddiv/kdd_lightaug.pkl,ckpt/kddiv/kdd_purekd.pkl,ckpt/kddiv/kdd_halfkd.pkl
until [ -f ckpt/kddiv/kdd_heavyaug.pkl ] && [ -f ckpt/kddiv/kdd_lightaug.pkl ] && [ -f ckpt/kddiv/kdd_purekd.pkl ] && [ -f ckpt/kddiv/kdd_halfkd.pkl ]; do sleep 300; done
sleep 30
for i in $(seq 0 23); do
  while [ "$(awk '{print ($1>105)?1:0}' /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f kd_blocks/kddiv4_b$i.json ]    || python3 e12_ens_gate.py --cand $D4 --ref $A --seeds 500 --workers 60 --seed0 $((360000 + i*1000)) --out kd_blocks/kddiv4_b$i.json
  [ -f kd_blocks/kdens7div_b$i.json ] || python3 e12_ens_gate.py --cand $K3,$D4 --ref $A --seeds 500 --workers 60 --seed0 $((360000 + i*1000)) --out kd_blocks/kdens7div_b$i.json
done
python3 - << "PYEOF"
import json, glob, sys, numpy as np
f3 = sorted(glob.glob("kd_blocks/kdconf_b*.json"))
out = {}; fail = []
v3 = np.array([json.load(open(f))["placement_pts"] for f in f3]) if len(f3) == 24 else None
for tag in ["kddiv4", "kdens7div"]:
    fs = sorted(glob.glob(f"kd_blocks/{tag}_b*.json"))
    if len(fs) != 24:
        fail.append(f"{tag}: {len(fs)}/24"); continue
    v = np.array([json.load(open(f))["placement_pts"] for f in fs])
    n = 24; se = v.std(ddof=1)/np.sqrt(n); lo = v.mean() - 2.069*se
    e = dict(n_blocks=n, mean=round(float(v.mean()),4), ci95_lo=round(float(lo),4),
             verdict="BEATS_AUGS0" if lo>2.5 else "TIED_NOT_SEPARATED")
    if v3 is not None:
        d = v - v3; sed = d.std(ddof=1)/np.sqrt(n); lod = d.mean() - 2.069*sed
        e["paired_vs_kdens3"] = dict(mean_diff=round(float(d.mean()),4), ci95_lo_diff=round(float(lod),4),
                                     verdict="BETTER_THAN_KDENS3" if lod>0 else "NOT_SEPARATED_FROM_KDENS3")
    out[tag] = e
out["integrity"] = "OK" if not fail else "LOUDFAIL: " + "; ".join(fail)
json.dump(out, open("KDDIV_RESULTS.json","w"), indent=2)
print(json.dumps(out, indent=2))
if fail: sys.exit(1)
PYEOF
echo "KDDIV DONE"
