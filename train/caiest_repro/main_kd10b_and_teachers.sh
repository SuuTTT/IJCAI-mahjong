#!/bin/bash
cd /root/IJCAI-mahjong/train/caiest_repro
# GPUs: 4 more teachers (s14-s17) for a potential 18-teacher rung
CUDA_VISIBLE_DEVICES=0 nohup python3 e11_train.py --channels 128 --blocks 40 --steps 90000 --seed 14 --out ckpt/aug/aug_128x40_s14.pkl > logs/e11_s14.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 e11_train.py --channels 128 --blocks 40 --steps 90000 --seed 15 --out ckpt/aug/aug_128x40_s15.pkl > logs/e11_s15.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 e11_train.py --channels 128 --blocks 40 --steps 90000 --seed 16 --out ckpt/aug/aug_128x40_s16.pkl > logs/e11_s16.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 e11_train.py --channels 128 --blocks 40 --steps 90000 --seed 17 --out ckpt/aug/aug_128x40_s17.pkl > logs/e11_s17.log 2>&1 &
# CPU: gate the main-box kd10 trio now (24 blocks, paired walls 360000+)
A=ckpt/aug/aug_128x40_s0.pkl
KB=ckpt/kd10/kd10_s8.pkl,ckpt/kd10/kd10_s9.pkl,ckpt/kd10/kd10_s10.pkl
for i in $(seq 0 23); do
  while [ "$(awk '{print ($1>105)?1:0}' /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f kd_blocks/kd10ensB_b$i.json ] || python3 e12_ens_gate.py --cand $KB --ref $A --seeds 500 --workers 58 --seed0 $((360000 + i*1000)) --out kd_blocks/kd10ensB_b$i.json
done
python3 - << "PYEOF"
import json, glob, numpy as np
f3 = sorted(glob.glob("kd_blocks/kdconf_b*.json"))
v3 = np.array([json.load(open(f))["placement_pts"] for f in f3])
fs = sorted(glob.glob("kd_blocks/kd10ensB_b*.json"))
v = np.array([json.load(open(f))["placement_pts"] for f in fs])
n = len(v); se = v.std(ddof=1)/np.sqrt(n); lo = v.mean() - 2.069*se
d = v - v3; sed = d.std(ddof=1)/np.sqrt(n); lod = d.mean() - 2.069*sed
out = dict(candidate="kd10ensB (main-box trio s38-41 seeds)", n_blocks=n,
           mean=round(float(v.mean()),4), ci95_lo=round(float(lo),4),
           verdict="BEATS_AUGS0" if lo>2.5 else "TIED_NOT_SEPARATED",
           paired_vs_kdens3=dict(mean_diff=round(float(d.mean()),4), ci95_lo_diff=round(float(lod),4),
                                 verdict="BETTER_THAN_KDENS3" if lod>0 else "NOT_SEPARATED"),
           integrity="OK" if n==24 else "LOUDFAIL n=%d" % n)
json.dump(out, open("KD10B_RESULTS.json","w"), indent=2)
print(json.dumps(out, indent=2))
PYEOF
echo KD10B_DONE
