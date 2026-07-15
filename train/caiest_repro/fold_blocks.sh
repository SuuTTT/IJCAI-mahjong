#!/bin/bash
# Overnight fold-config block campaign. Args: <box-tag> <config-list>
# configs: name:sh_fold:sh_dead:min_turn:min_melds
cd $REPRO
K3=ckpt/kd/kd_128x40_s0.pkl,ckpt/kd/kd_128x40_s1.pkl,ckpt/kd/kd_128x40_s2.pkl
mkdir -p fold_blocks
for cfg in $CFGS; do
  IFS=: read NAME SF SD MT MM <<< "$cfg"
  for i in $(seq 0 11); do
    [ -f fold_blocks/${NAME}_b$i.json ] || python3 e20_fold_gate.py --cand $K3 --ref $K3 \
      --sh-fold $SF --sh-dead $SD --min-turn $MT --min-melds $MM \
      --seeds 500 --workers $WK --seed0 $((820000 + i*1000)) --out fold_blocks/${NAME}_b$i.json
  done
done
echo FOLD_BLOCKS_DONE_$TAG
