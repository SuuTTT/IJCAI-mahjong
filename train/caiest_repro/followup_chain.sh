#!/bin/bash
# 07-04 follow-ups: mix6 to 24 blocks; gen-2 KD gates when trainers finish.
cd /root/IJCAI-mahjong/train/caiest_repro
A=ckpt/aug/aug_128x40_s0.pkl
K0=ckpt/kd/kd_128x40_s0.pkl; K1=ckpt/kd/kd_128x40_s1.pkl; K2=ckpt/kd/kd_128x40_s2.pkl
G0=ckpt/kdcap/kd_192x40_s0.pkl; G1=ckpt/kdcap/kd_192x40_s1.pkl; G2=ckpt/kdcap/kd_192x40_s2.pkl
for i in $(seq 12 23); do
  while [ "$(awk '{print ($1>105)?1:0}' /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f kd_blocks/mix6_b$i.json ] || python3 e12_ens_gate.py --cand $K0,$K1,$K2,$G0@192,$G1@192,$G2@192 --ref $A --seeds 500 --workers 60 --seed0 $((440000 + i*1000)) --out kd_blocks/mix6_b$i.json
done
until [ -f ckpt/kd2/kd2_128x40_s0.pkl ] && [ -f ckpt/kd2/kd2_128x40_s1.pkl ] && [ -f ckpt/kd2/kd2_128x40_s2.pkl ]; do sleep 300; done
sleep 30
D0=ckpt/kd2/kd2_128x40_s0.pkl; D1=ckpt/kd2/kd2_128x40_s1.pkl; D2=ckpt/kd2/kd2_128x40_s2.pkl
for i in $(seq 0 5); do
  while [ "$(awk '{print ($1>105)?1:0}' /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f kd_blocks/kd2s0_b$i.json ] || python3 e8_gate.py --cand $D0 --ref $A --lam 0 --seeds 500 --workers 60 --seed0 $((460000 + i*1000)) --out kd_blocks/kd2s0_b$i.json
  [ -f kd_blocks/kd2s1_b$i.json ] || python3 e8_gate.py --cand $D1 --ref $A --lam 0 --seeds 500 --workers 60 --seed0 $((470000 + i*1000)) --out kd_blocks/kd2s1_b$i.json
  [ -f kd_blocks/kd2s2_b$i.json ] || python3 e8_gate.py --cand $D2 --ref $A --lam 0 --seeds 500 --workers 60 --seed0 $((480000 + i*1000)) --out kd_blocks/kd2s2_b$i.json
done
for i in $(seq 0 11); do
  while [ "$(awk '{print ($1>105)?1:0}' /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f kd_blocks/kd2ens_b$i.json ] || python3 e12_ens_gate.py --cand $D0,$D1,$D2 --ref $A --seeds 500 --workers 60 --seed0 $((490000 + i*1000)) --out kd_blocks/kd2ens_b$i.json
done
python3 - << "PYEOF"
import json, glob, sys, numpy as np
T = {5: 2.571, 11: 2.201, 23: 2.069}
out = {}; fail = []
for tag, want in [("mix6", 24), ("kd2s0", 6), ("kd2s1", 6), ("kd2s2", 6), ("kd2ens", 12)]:
    fs = sorted(glob.glob(f"kd_blocks/{tag}_b*.json"))
    if len(fs) != want:
        fail.append(f"{tag}: expected {want}, found {len(fs)}"); continue
    v = np.array([json.load(open(f))["placement_pts"] for f in fs])
    n = len(v); se = v.std(ddof=1)/np.sqrt(n); lo = v.mean() - T.get(n-1, 2.201)*se
    out[tag] = dict(n_blocks=n, mean=round(float(v.mean()),4), ci95_lo=round(float(lo),4),
                    margin_lo=round(float(lo-2.5),4), blocks=[round(float(x),4) for x in v],
                    verdict="BEATS_AUGS0" if lo>2.5 else "TIED_NOT_SEPARATED")
out["integrity"] = "OK" if not fail else "LOUDFAIL: " + "; ".join(fail)
json.dump(out, open("FOLLOWUP_RESULTS.json","w"), indent=2)
print(json.dumps(out, indent=2))
if fail: sys.exit(1)
PYEOF
echo "FOLLOWUP DONE"
