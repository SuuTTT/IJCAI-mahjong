#!/bin/bash
# 8h autonomous danger/field-clone program (main box).
# 1) wait for teacher GPUs to free  2) train field-clone (GPU0) + danger head (GPU1)
# 3) e17 calibration + margin sweep vs field-clone opponents  4) aggregate
cd /root/IJCAI-mahjong/train/caiest_repro
until [ -f ckpt/aug/aug_128x40_s14.pkl ] && [ -f ckpt/aug/aug_128x40_s15.pkl ] && [ -f ckpt/aug/aug_128x40_s16.pkl ] && [ -f ckpt/aug/aug_128x40_s17.pkl ]; do sleep 300; done
sleep 30
mkdir -p ckpt/danger
CUDA_VISIBLE_DEVICES=0 nohup python3 e18_finetune.py --channels 128 --blocks 40 --steps 20000 --lr 5e-5 --seed 70 \
  --init ckpt/aug/aug_128x40_s0.bn.pkl --data /root/IJCAI-mahjong/data/processed/sim11_danger_bc.npz \
  --out ckpt/danger/fieldclone.pkl > logs/e18_fieldclone.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 e19_danger.py --channels 128 --blocks 40 --steps 15000 --lr 1e-4 --seed 71 \
  --valevery 1000000 --init ckpt/aug/aug_128x40_s0.bn.pkl --data /root/IJCAI-mahjong/data/processed/sim11_danger.npz \
  --out ckpt/danger/danger235.pkl > logs/e19_danger.log 2>&1 &
until [ -f ckpt/danger/fieldclone.pkl ] && [ -f ckpt/danger/danger235.pkl ]; do sleep 300; done
sleep 30
K3=ckpt/kd/kd_128x40_s0.pkl,ckpt/kd/kd_128x40_s1.pkl,ckpt/kd/kd_128x40_s2.pkl
FC=ckpt/danger/fieldclone.pkl
DG=ckpt/danger/danger235.pkl
mkdir -p danger_blocks
# calibration: danger-off => diff must be exactly 0
python3 e17_danger_gate.py --cand $K3 --opp $FC --danger $DG --danger-off --seeds 60 --workers 60 --seed0 700000 --out danger_blocks/calib.json
# margin sweep (short blocks)
for MG in 0.5 1.0 2.0; do
  python3 e17_danger_gate.py --cand $K3 --opp $FC --danger $DG --margin $MG --seeds 120 --workers 60 --seed0 701000 --out danger_blocks/sweep_m$MG.json
done
# pick best margin by diff, then 12 full blocks at that margin
BEST=$(python3 -c "
import json, glob
xs = [(json.load(open(f))[\"diff\"], json.load(open(f))[\"margin\"]) for f in glob.glob(\"danger_blocks/sweep_m*.json\")]
print(sorted(xs)[-1][1])")
echo "BEST_MARGIN=$BEST"
for i in $(seq 0 11); do
  while [ "$(awk '{print ($1>105)?1:0}' /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f danger_blocks/full_b$i.json ] || python3 e17_danger_gate.py --cand $K3 --opp $FC --danger $DG --margin $BEST --seeds 500 --workers 60 --seed0 $((710000 + i*1000)) --out danger_blocks/full_b$i.json
done
python3 - << "PYEOF"
import json, glob, numpy as np
fs = sorted(glob.glob("danger_blocks/full_b*.json"))
ds = np.array([json.load(open(f))["diff"] for f in fs])
dinP = sum(json.load(open(f))["dealins_plain"] for f in fs)
dinD = sum(json.load(open(f))["dealins_danger"] for f in fs)
n = len(ds); se = ds.std(ddof=1)/np.sqrt(n); lo = ds.mean() - 2.201*se
out = dict(n_blocks=n, mean_diff=round(float(ds.mean()),4), ci95_lo=round(float(lo),4),
           dealins_plain=dinP, dealins_danger=dinD,
           verdict="DANGER_HELPS" if lo > 0 else "NOT_SEPARATED",
           integrity="OK" if n == 12 else "LOUDFAIL n=%d" % n)
json.dump(out, open("DANGER_RESULTS.json","w"), indent=2)
print(json.dumps(out, indent=2))
PYEOF
echo DANGER_CHAIN_DONE
