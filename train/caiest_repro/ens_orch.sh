#!/bin/bash
cd /root/IJCAI-mahjong/train/caiest_repro
A=ckpt/aug/aug_128x40_s0.pkl; B=ckpt/aug/aug_128x40_s1.pkl; C=ckpt/aug/aug_128x40_s2.pkl
D=ckpt/e1b/full_128x40_s1.pkl
mkdir -p ens_blocks
for i in $(seq 0 11); do
  S0=$((210000 + i*1000))
  while [ "$(awk "{print (\$1>105)?1:0}" /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f ens_blocks/ens3_b$i.json ] || python3 e12_ens_gate.py --cand $A,$B,$C    --ref $A --seeds 500 --workers 64 --seed0 $S0 --out ens_blocks/ens3_b$i.json
  [ -f ens_blocks/ens4_b$i.json ] || python3 e12_ens_gate.py --cand $A,$B,$C,$D --ref $A --seeds 500 --workers 64 --seed0 $S0 --out ens_blocks/ens4_b$i.json
done
python3 - << "PYEOF"
import json, glob, numpy as np
out = {"experiment": "SEED-ENSEMBLE (deploy mean-softmax rule) vs aug_s0", "ref": "aug_128x40_s0.pkl",
       "rule": "beat iff 95% block-t CI lower bound > 2.500", "variants": {}}
for tag in ["ens3", "ens4"]:
    vals = [json.load(open(f))["placement_pts"] for f in sorted(glob.glob(f"ens_blocks/{tag}_b*.json"))]
    v = np.array(vals); n = len(v)
    se = v.std(ddof=1)/np.sqrt(n); tcrit = {11:2.201,12:2.179}.get(n-1, 2.2)
    lo, hi = v.mean()-tcrit*se, v.mean()+tcrit*se
    out["variants"][tag] = dict(n_blocks=n, games=n*2000, mean=round(float(v.mean()),4),
        sd=round(float(v.std(ddof=1)),4), ci95_lo=round(float(lo),4), ci95_hi=round(float(hi),4),
        margin_lo=round(float(lo-2.5),4), blocks=[round(float(x),4) for x in vals],
        verdict="BEATS_AUGS0" if lo > 2.5 else "TIED_NOT_SEPARATED")
json.dump(out, open("ENSEMBLE_RESULTS.json","w"), indent=2)
print(json.dumps(out, indent=2))
PYEOF
echo "ENS ORCH ALL DONE"
