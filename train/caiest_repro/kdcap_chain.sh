#!/bin/bash
# Overnight chain 2026-07-03:
# 1) wait kd192 s0/s1/s2 -> relaunch GPUs 0/1/3 with gen-2 KD (9 teachers: 3xKD + 6xaug)
# 2) gate kd192 singles (6 blocks each) + kd192-ens (12) + mixed 6-model ens (12)
# 3) wait kd256 -> gen-2 seed3 on GPU2, gate kd256 single (6 blocks)
# 4) aggregate -> KDCAP_RESULTS.json (loud-fail)
cd /root/IJCAI-mahjong/train/caiest_repro
A=ckpt/aug/aug_128x40_s0.pkl
K0=ckpt/kd/kd_128x40_s0.pkl; K1=ckpt/kd/kd_128x40_s1.pkl; K2=ckpt/kd/kd_128x40_s2.pkl
G0=ckpt/kdcap/kd_192x40_s0.pkl; G1=ckpt/kdcap/kd_192x40_s1.pkl; G2=ckpt/kdcap/kd_192x40_s2.pkl
C0=ckpt/kdcap/kd_256x40_s0.pkl
T9=$K0,$K1,$K2,ckpt/aug/aug_128x40_s0.pkl,ckpt/aug/aug_128x40_s1.pkl,ckpt/aug/aug_128x40_s2.pkl,ckpt/aug/aug_128x40_s3.pkl,ckpt/aug/aug_128x40_s4.pkl,ckpt/aug/aug_128x40_s5.pkl
mkdir -p ckpt/kd2 kd_blocks

until [ -f $G0 ] && [ -f $G1 ] && [ -f $G2 ]; do sleep 300; done
sleep 60
CUDA_VISIBLE_DEVICES=0 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 10 --teachers $T9 --alpha 0.7 --out ckpt/kd2/kd2_128x40_s0.pkl > logs/e13_kd2_s0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 11 --teachers $T9 --alpha 0.7 --out ckpt/kd2/kd2_128x40_s1.pkl > logs/e13_kd2_s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 12 --teachers $T9 --alpha 0.7 --out ckpt/kd2/kd2_128x40_s2.pkl > logs/e13_kd2_s2.log 2>&1 &

for i in $(seq 0 5); do
  while [ "$(awk '{print ($1>105)?1:0}' /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f kd_blocks/g192s0_b$i.json ] || python3 e12_ens_gate.py --cand $G0@192 --ref $A --seeds 500 --workers 55 --seed0 $((400000 + i*1000)) --out kd_blocks/g192s0_b$i.json
  [ -f kd_blocks/g192s1_b$i.json ] || python3 e12_ens_gate.py --cand $G1@192 --ref $A --seeds 500 --workers 55 --seed0 $((410000 + i*1000)) --out kd_blocks/g192s1_b$i.json
  [ -f kd_blocks/g192s2_b$i.json ] || python3 e12_ens_gate.py --cand $G2@192 --ref $A --seeds 500 --workers 55 --seed0 $((420000 + i*1000)) --out kd_blocks/g192s2_b$i.json
done
for i in $(seq 0 11); do
  while [ "$(awk '{print ($1>105)?1:0}' /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f kd_blocks/g192ens_b$i.json ] || python3 e12_ens_gate.py --cand $G0@192,$G1@192,$G2@192 --ref $A --seeds 500 --workers 55 --seed0 $((430000 + i*1000)) --out kd_blocks/g192ens_b$i.json
  [ -f kd_blocks/mix6_b$i.json ]   || python3 e12_ens_gate.py --cand $K0,$K1,$K2,$G0@192,$G1@192,$G2@192 --ref $A --seeds 500 --workers 55 --seed0 $((440000 + i*1000)) --out kd_blocks/mix6_b$i.json
done

until [ -f $C0 ]; do sleep 300; done
sleep 60
CUDA_VISIBLE_DEVICES=2 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 13 --teachers $T9 --alpha 0.7 --out ckpt/kd2/kd2_128x40_s3.pkl > logs/e13_kd2_s3.log 2>&1 &
for i in $(seq 0 5); do
  while [ "$(awk '{print ($1>105)?1:0}' /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f kd_blocks/c256_b$i.json ] || python3 e12_ens_gate.py --cand $C0@256 --ref $A --seeds 500 --workers 55 --seed0 $((450000 + i*1000)) --out kd_blocks/c256_b$i.json
done

python3 - << "PYEOF"
import json, glob, sys, numpy as np
T = {5: 2.571, 11: 2.201, 23: 2.069}
out = {}; fail = []
for tag, want in [("g192s0",6),("g192s1",6),("g192s2",6),("g192ens",12),("mix6",12),("c256",6)]:
    fs = sorted(glob.glob(f"kd_blocks/{tag}_b*.json"))
    if len(fs) != want:
        fail.append(f"{tag}: expected {want}, found {len(fs)}"); continue
    v = np.array([json.load(open(f))["placement_pts"] for f in fs])
    n = len(v); se = v.std(ddof=1)/np.sqrt(n); lo = v.mean() - T.get(n-1, 2.201)*se
    out[tag] = dict(n_blocks=n, mean=round(float(v.mean()),4), ci95_lo=round(float(lo),4),
                    margin_lo=round(float(lo-2.5),4), blocks=[round(float(x),4) for x in v],
                    verdict="BEATS_AUGS0" if lo>2.5 else "TIED_NOT_SEPARATED")
out["integrity"] = "OK" if not fail else "LOUDFAIL: " + "; ".join(fail)
json.dump(out, open("KDCAP_RESULTS.json","w"), indent=2)
print(json.dumps(out, indent=2))
if fail: sys.exit(1)
PYEOF
echo "KDCAP CHAIN DONE"
