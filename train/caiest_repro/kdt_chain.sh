#!/bin/bash
# Temperature-KD program (2026-07-04 overnight-2):
# waits for diversity students to finish -> 4 T-students on the 4 GPUs ->
# prespecified gates: kdT4ens (4 new alone) + kdens7T (kdens3 + 4 new),
# 24 blocks each at seed0 360000+ (paired vs kdconf) -> KDT_RESULTS.json
cd /root/IJCAI-mahjong/train/caiest_repro
A=ckpt/aug/aug_128x40_s0.pkl
T6=ckpt/aug/aug_128x40_s0.pkl,ckpt/aug/aug_128x40_s1.pkl,ckpt/aug/aug_128x40_s2.pkl,ckpt/aug/aug_128x40_s3.pkl,ckpt/aug/aug_128x40_s4.pkl,ckpt/aug/aug_128x40_s5.pkl
K3=ckpt/kd/kd_128x40_s0.pkl,ckpt/kd/kd_128x40_s1.pkl,ckpt/kd/kd_128x40_s2.pkl
mkdir -p ckpt/kdt
until [ -f ckpt/kddiv/kdd_heavyaug.pkl ] && [ -f ckpt/kddiv/kdd_lightaug.pkl ] && [ -f ckpt/kddiv/kdd_purekd.pkl ] && [ -f ckpt/kddiv/kdd_halfkd.pkl ]; do sleep 300; done
sleep 60
CUDA_VISIBLE_DEVICES=0 nohup python3 e13b_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 10 --teachers $T6 --alpha 0.7 --temp 2.0 --out ckpt/kdt/kdt_T2.pkl > logs/e13b_T2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 e13b_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 11 --teachers $T6 --alpha 0.7 --temp 4.0 --out ckpt/kdt/kdt_T4.pkl > logs/e13b_T4.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 e13b_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 12 --teachers $T6 --alpha 0.7 --temp 2.0 --p_suit 0.6 --p_ref 0.3 --p_drag 0.3 --out ckpt/kdt/kdt_T2light.pkl > logs/e13b_T2light.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 e13b_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 13 --teachers $T6 --alpha 1.0 --temp 2.0 --out ckpt/kdt/kdt_T2pure.pkl > logs/e13b_T2pure.log 2>&1 &
until [ -f ckpt/kdt/kdt_T2.pkl ] && [ -f ckpt/kdt/kdt_T4.pkl ] && [ -f ckpt/kdt/kdt_T2light.pkl ] && [ -f ckpt/kdt/kdt_T2pure.pkl ]; do sleep 600; done
T4E=ckpt/kdt/kdt_T2.pkl,ckpt/kdt/kdt_T4.pkl,ckpt/kdt/kdt_T2light.pkl,ckpt/kdt/kdt_T2pure.pkl
for i in $(seq 0 23); do
  while [ "$(awk '{print ($1>105)?1:0}' /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f kd_blocks/kdT4ens_b$i.json ] || python3 e12_ens_gate.py --cand $T4E --ref $A --seeds 500 --workers 60 --seed0 $((360000 + i*1000)) --out kd_blocks/kdT4ens_b$i.json
  [ -f kd_blocks/kdens7T_b$i.json ] || python3 e12_ens_gate.py --cand $K3,$T4E --ref $A --seeds 500 --workers 60 --seed0 $((360000 + i*1000)) --out kd_blocks/kdens7T_b$i.json
done
python3 - << "PYEOF"
import json, glob, sys, numpy as np
f3 = sorted(glob.glob("kd_blocks/kdconf_b*.json"))
v3 = np.array([json.load(open(f))["placement_pts"] for f in f3]) if len(f3) == 24 else None
out = {}; fail = []
for tag in ["kdT4ens", "kdens7T"]:
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
json.dump(out, open("KDT_RESULTS.json","w"), indent=2)
print(json.dumps(out, indent=2))
if fail: sys.exit(1)
PYEOF
echo "KDT CHAIN DONE"
