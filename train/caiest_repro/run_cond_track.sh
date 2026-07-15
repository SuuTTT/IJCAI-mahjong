#!/bin/bash
cd /root/caiest_repro
run_gpu () { local g=$1; shift
  for cmd in "$@"; do CUDA_VISIBLE_DEVICES=$g bash -c "$cmd" >> logs/cond_g$g.log 2>&1; done; }
J () { echo "[ -f ckpt/cond/$1.pkl ] || python3 e11_cond_train.py --plane $2 --seed $3 --steps 60000 --out ckpt/cond/$1.pkl"; }
run_gpu 2 "$(J cond_s0 src 0)"  "$(J zero_s1 zero 1)" &
run_gpu 3 "$(J cond_s1 src 1)"  "$(J zero_s2 zero 2)" &
run_gpu 4 "$(J zero_s0 zero 0)" "$(J cond_s2 src 2)" &
wait
touch results/COND_TRAIN_DONE; echo COND_TRAIN_DONE
