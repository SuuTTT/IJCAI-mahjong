#!/bin/bash
# Placement-gate the sample-eff KD students by data fraction vs aug_s0 (calibrated 2.500).
# Tests whether the val-acc sample-efficiency curve holds on the real 2.5 placement metric.
cd /root/caiest_repro
A=ckpt/aug/aug_128x40_s0.pkl
mkdir -p sampeff_gate
for f in 002 005 010 025 050 100; do
  C=$(ls ckpt/sampeff/kdf_f${f}_s*.pkl 2>/dev/null | head -3 | tr '\n' ',' | sed 's/,$//')
  [ -z "$C" ] && continue
  for i in 0 1 2 3 4 5; do
    while [ "$(awk '{print ($1>110)?1:0}' /proc/loadavg)" = "1" ]; do sleep 60; done
    [ -f sampeff_gate/f${f}_b$i.json ] || python3 e12_ens_gate.py --cand $C --ref $A --seeds 500 --workers 40 --seed0 $((500000 + i*1000)) --out sampeff_gate/f${f}_b$i.json
  done
done
python3 - << 'PYEOF'
import json, glob, numpy as np
print("=== SAMPLE-EFF PLACEMENT (ensemble of KD students at each data fraction, vs aug_s0=2.500) ===")
out={}
for f in ["002","005","010","025","050","100"]:
    fs=sorted(glob.glob("sampeff_gate/f%s_b*.json"%f))
    if not fs: continue
    v=np.array([json.load(open(x))["placement_pts"] for x in fs])
    n=len(v); se=v.std(ddof=1)/np.sqrt(n); lo=v.mean()-2.571*se
    out[f]=dict(n=n, mean=round(float(v.mean()),4), ci_lo=round(float(lo),4))
    print("frac %-4s : placement=%.4f  ci_lo=%.4f  %s"%(f, v.mean(), lo, "BEATS_aug" if lo>2.5 else "tie/below"))
json.dump(out, open("SAMPEFF_PLACEMENT.json","w"), indent=2)
PYEOF
echo SAMPEFF_PLACEMENT_DONE
