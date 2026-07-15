#!/bin/bash
# Track 2: CIFAR-100N intermediate REAL-noise curve (mix25/50/75 = subsampled human-noisy labels)
cd /root/e1_cifarn || exit 1
mkdir -p logs results ckpt
run_gpu () { local g=$1; shift
  for lv in "$@"; do
    [ -f "results/e1_c100_${lv}.json" ] && continue
    echo "[mix] c100_$lv -> GPU $g start $(date)"
    python3 cifarn_e1.py --dataset c100 --noise "$lv" --gpu "$g" --epochs 60 --data data >> "logs/c100_${lv}.log" 2>&1
    echo "[mix] c100_$lv exit $? $(date)"
  done; }
run_gpu 5 mix25 mix75 &
run_gpu 6 mix50 &
wait
python3 aggregate.py > logs/aggregate_c100_mix.log 2>&1
touch results/E1_C100_MIX_DONE; echo E1_C100_MIX_DONE
