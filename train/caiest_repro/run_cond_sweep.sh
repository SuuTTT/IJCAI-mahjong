#!/bin/bash
# f2_frac sweep + extra seed for the source-conditioned 2027 candidate.
cd /root/caiest_repro
J () { echo "[ -f ckpt/cond/$1.pkl ] || python3 e11_cond_train.py --plane src --seed $2 --f2_frac $3 --steps 60000 --out ckpt/cond/$1.pkl"; }
run () { local g=$1; shift; for c in "$@"; do CUDA_VISIBLE_DEVICES=$g bash -c "$c" >> logs/cond_sweep_g$g.log 2>&1; done; }
run 0 "$(J cond_f035_s0 0 0.35)" &
run 2 "$(J cond_f065_s0 0 0.65)" &
run 3 "$(J cond_f080_s0 0 0.80)" &
run 7 "$(J cond_s3 3 0.5)" &
wait
touch results/COND_SWEEP_DONE; echo COND_SWEEP_DONE
