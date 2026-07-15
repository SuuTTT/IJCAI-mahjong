#!/bin/bash
cd /root/caiest_repro
mkdir -p ckpt/oppbelief
J(){ local g=$1 s=$2 ch=$3 bl=$4 out=$5
  [ -f "$out" ] && return
  CUDA_VISIBLE_DEVICES=$g python3 oppbelief_train.py --tag full --seed $s --steps 50000 --channels $ch --blocks $bl --out $out >> logs/belief_extra_g$g.log 2>&1; }
# stronger 192x48 predictor (2 seeds) + extend base 128x40 ensemble (3 more seeds)
J 3 10 192 48 ckpt/oppbelief/oppbelief_big_s10.pt &
J 4 11 192 48 ckpt/oppbelief/oppbelief_big_s11.pt &
J 5 3 128 40 ckpt/oppbelief/oppbelief_s3.pt &
J 6 4 128 40 ckpt/oppbelief/oppbelief_s4.pt &
J 7 5 128 40 ckpt/oppbelief/oppbelief_s5.pt &
wait; touch results/BELIEF_EXTRA_DONE; echo BELIEF_EXTRA_DONE
