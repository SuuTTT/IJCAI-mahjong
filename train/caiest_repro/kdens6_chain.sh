#!/bin/bash
# Scale the winning axis: +3 gen-1-recipe KD students (seeds 3/4/5) -> kdens6.
# Gate kdens6 at the SAME seed0 range as KDENS_CONFIRM (360000+) for a paired
# per-block comparison vs kdens3. Waits for gen-2 trainers to free GPUs 0/1/3.
cd /root/IJCAI-mahjong/train/caiest_repro
A=ckpt/aug/aug_128x40_s0.pkl
T6=ckpt/aug/aug_128x40_s0.pkl,ckpt/aug/aug_128x40_s1.pkl,ckpt/aug/aug_128x40_s2.pkl,ckpt/aug/aug_128x40_s3.pkl,ckpt/aug/aug_128x40_s4.pkl,ckpt/aug/aug_128x40_s5.pkl
until [ -f ckpt/kd2/kd2_128x40_s0.pkl ] && [ -f ckpt/kd2/kd2_128x40_s1.pkl ] && [ -f ckpt/kd2/kd2_128x40_s2.pkl ]; do sleep 300; done
sleep 30
CUDA_VISIBLE_DEVICES=0 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 3 --teachers $T6 --alpha 0.7 --out ckpt/kd/kd_128x40_s3.pkl > logs/e13_kd_s3.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 4 --teachers $T6 --alpha 0.7 --out ckpt/kd/kd_128x40_s4.pkl > logs/e13_kd_s4.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 5 --teachers $T6 --alpha 0.7 --out ckpt/kd/kd_128x40_s5.pkl > logs/e13_kd_s5.log 2>&1 &
until [ -f ckpt/kd/kd_128x40_s3.pkl ] && [ -f ckpt/kd/kd_128x40_s4.pkl ] && [ -f ckpt/kd/kd_128x40_s5.pkl ]; do sleep 600; done
K=ckpt/kd/kd_128x40_s0.pkl,ckpt/kd/kd_128x40_s1.pkl,ckpt/kd/kd_128x40_s2.pkl,ckpt/kd/kd_128x40_s3.pkl,ckpt/kd/kd_128x40_s4.pkl,ckpt/kd/kd_128x40_s5.pkl
for i in $(seq 0 23); do
  while [ "$(awk '{print ($1>105)?1:0}' /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f kd_blocks/kdens6_b$i.json ] || python3 e12_ens_gate.py --cand $K --ref $A --seeds 500 --workers 60 --seed0 $((360000 + i*1000)) --out kd_blocks/kdens6_b$i.json
done
python3 - << "PYEOF"
import json, glob, sys, numpy as np
f6 = sorted(glob.glob("kd_blocks/kdens6_b*.json"))
f3 = sorted(glob.glob("kd_blocks/kdconf_b*.json"))
out = {}
if len(f6) != 24 or len(f3) != 24:
    out["integrity"] = "LOUDFAIL: kdens6=%d kdconf=%d (want 24 each)" % (len(f6), len(f3))
else:
    v6 = np.array([json.load(open(f))["placement_pts"] for f in f6])
    v3 = np.array([json.load(open(f))["placement_pts"] for f in f3])
    n = 24
    se6 = v6.std(ddof=1)/np.sqrt(n); lo6 = v6.mean() - 2.069*se6
    d = v6 - v3   # paired per-block (same seed0 range)
    sed = d.std(ddof=1)/np.sqrt(n); lod = d.mean() - 2.069*sed
    out = dict(
        kdens6=dict(n_blocks=n, mean=round(float(v6.mean()),4), ci95_lo=round(float(lo6),4),
                    verdict="BEATS_AUGS0" if lo6>2.5 else "TIED_NOT_SEPARATED",
                    blocks=[round(float(x),4) for x in v6]),
        paired_vs_kdens3=dict(mean_diff=round(float(d.mean()),4), ci95_lo_diff=round(float(lod),4),
                              verdict="KDENS6_BETTER" if lod>0 else "NOT_SEPARATED_FROM_KDENS3"),
        integrity="OK")
json.dump(out, open("KDENS6_RESULTS.json","w"), indent=2)
print(json.dumps(out, indent=2))
if out.get("integrity") != "OK": sys.exit(1)
PYEOF
echo "KDENS6 CHAIN DONE"
