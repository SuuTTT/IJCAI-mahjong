#!/bin/bash
# Fully autonomous chain: wait gen -> train 3 seeds (GPU3: s0->s2, GPU4: s1) -> verdict eval.
cd /root/caiest_repro
mkdir -p logs ckpt/dealin_pc_v2 results
exec > logs/oppdealin_orchestrate.log 2>&1
STEPS=50000
DATA=data/oppdealin/full
OUT=ckpt/dealin_pc_v2

echo "[orch] $(date) waiting for gen-complete marker"
while [ ! -f logs/oppdealin_full_gen.DONE ]; do sleep 30; done
echo "[orch] $(date) gen done -> launching training on GPUs 3,4"

# GPU4: seed 1
setsid bash -c "CUDA_VISIBLE_DEVICES=4 python3 oppdealin_train.py --seed 1 --steps $STEPS \
  --data $DATA --out $OUT/dealin_pc_v2_s1.pt > logs/oppdealin_train_s1.log 2>&1; \
  touch logs/oppdealin_train_s1.DONE" < /dev/null &

# GPU3: seed 0 then seed 2
setsid bash -c "CUDA_VISIBLE_DEVICES=3 python3 oppdealin_train.py --seed 0 --steps $STEPS \
  --data $DATA --out $OUT/dealin_pc_v2_s0.pt > logs/oppdealin_train_s0.log 2>&1; \
  touch logs/oppdealin_train_s0.DONE; \
  CUDA_VISIBLE_DEVICES=3 python3 oppdealin_train.py --seed 2 --steps $STEPS \
  --data $DATA --out $OUT/dealin_pc_v2_s2.pt > logs/oppdealin_train_s2.log 2>&1; \
  touch logs/oppdealin_train_s2.DONE" < /dev/null &

echo "[orch] $(date) training launched (GPU3=s0->s2, GPU4=s1); waiting for 3 DONE markers"
while [ ! -f logs/oppdealin_train_s0.DONE ] || [ ! -f logs/oppdealin_train_s1.DONE ] \
      || [ ! -f logs/oppdealin_train_s2.DONE ]; do sleep 60; done
echo "[orch] $(date) all 3 seeds done -> running v2-vs-on-policy verdict eval on GPU3"

CUDA_VISIBLE_DEVICES=3 python3 oppdealin_eval.py --data $DATA --v2 $OUT \
  --onpolicy ckpt/dealin_pc --out results/oppdealin_verdict.json > logs/oppdealin_eval.log 2>&1
touch logs/oppdealin_ALL_DONE
echo "[orch] $(date) COMPLETE -> ckpt/dealin_pc_v2/ + results/oppdealin_verdict.json"
