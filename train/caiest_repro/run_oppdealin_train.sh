#!/bin/bash
# 3-seed OFF-POLICY per-candidate deal-in (v2) training on GPUs 3,4.
# GPU3: seed0 then seed2 (chained). GPU4: seed1. Then the verdict eval on GPU3.
set -e
cd /root/caiest_repro
DATA=data/oppdealin/full
OUT=ckpt/dealin_pc_v2
mkdir -p $OUT logs
STEPS=50000

# GPU4: seed 1
setsid bash -c "CUDA_VISIBLE_DEVICES=4 python3 oppdealin_train.py --seed 1 --steps $STEPS \
  --data $DATA --out $OUT/dealin_pc_v2_s1.pt > logs/oppdealin_train_s1.log 2>&1; \
  touch logs/oppdealin_train_s1.DONE" < /dev/null &

# GPU3: seed 0, then seed 2
setsid bash -c "CUDA_VISIBLE_DEVICES=3 python3 oppdealin_train.py --seed 0 --steps $STEPS \
  --data $DATA --out $OUT/dealin_pc_v2_s0.pt > logs/oppdealin_train_s0.log 2>&1; \
  touch logs/oppdealin_train_s0.DONE; \
  CUDA_VISIBLE_DEVICES=3 python3 oppdealin_train.py --seed 2 --steps $STEPS \
  --data $DATA --out $OUT/dealin_pc_v2_s2.pt > logs/oppdealin_train_s2.log 2>&1; \
  touch logs/oppdealin_train_s2.DONE" < /dev/null &

echo "launched: GPU3=s0->s2, GPU4=s1"
