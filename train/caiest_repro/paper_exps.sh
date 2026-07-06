#!/bin/bash
# Paper experiments (GPU): alpha ablation + disjoint-teacher students.
# Env: REPRO, JOBS="name:alpha:teacherspec:seed:gpu ..."
cd $REPRO
T6=ckpt/aug/aug_128x40_s0.pkl,ckpt/aug/aug_128x40_s1.pkl,ckpt/aug/aug_128x40_s2.pkl,ckpt/aug/aug_128x40_s3.pkl,ckpt/aug/aug_128x40_s4.pkl,ckpt/aug/aug_128x40_s5.pkl
TA=ckpt/aug/aug_128x40_s0.pkl,ckpt/aug/aug_128x40_s1.pkl,ckpt/aug/aug_128x40_s2.pkl
TB=ckpt/aug/aug_128x40_s3.pkl,ckpt/aug/aug_128x40_s4.pkl,ckpt/aug/aug_128x40_s5.pkl
mkdir -p ckpt/paperx logs
for j in $JOBS; do
  IFS=: read NAME AL TS SEED GPU <<< "$j"
  case $TS in T6) T=$T6;; TA) T=$TA;; TB) T=$TB;; esac
  CUDA_VISIBLE_DEVICES=$GPU nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 \
    --seed $SEED --teachers $T --alpha $AL --out ckpt/paperx/$NAME.pkl > logs/px_$NAME.log 2>&1 &
done
wait
echo PAPER_EXPS_DONE
