#!/bin/bash
# Source-conditioned KD-ensemble students: seeds 0/1/2 on GPUs 4/5/6.
cd /root/caiest_repro
mkdir -p logs ckpt/cond_kd results

run_one () {
  gpu=$1; seed=$2
  out=ckpt/cond_kd/ckd_s${seed}.pkl
  if [ -f "$out" ]; then
    echo "skip seed$seed ($out already exists)"
    return
  fi
  CUDA_VISIBLE_DEVICES=$gpu python3 e11_cond_kd_train.py \
      --steps 60000 --alpha 0.7 --f2_frac 0.5 --seed "$seed" \
      --out "$out" > "logs/cond_kd_g${gpu}.log" 2>&1
}

run_one 4 0 &
run_one 5 1 &
run_one 6 2 &
wait
touch results/COND_KD_DONE
echo "ALL COND_KD DONE"
