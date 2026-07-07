#!/bin/bash
# Overnight: alpha-ablation ensemble gates on main (a03/a05/a09/a10 trios), 12 blocks each.
cd /root/IJCAI-mahjong/train/caiest_repro
A=ckpt/aug/aug_128x40_s0.pkl
mkdir -p kd_blocks
gate() {
  local NAME=$1 C=$2
  for i in $(seq 0 11); do
    [ -f kd_blocks/${NAME}_b$i.json ] || python3 e12_ens_gate.py --cand $C --ref $A --seeds 500 --workers 60 --seed0 $((360000 + i*1000)) --out kd_blocks/${NAME}_b$i.json
  done
}
gate a03ens ckpt/paperx/a03_s0.pkl,ckpt/paperx/a03_s1.pkl,ckpt/paperx/a03_s2.pkl
gate halfens ckpt/paperx/half_s0.pkl,ckpt/paperx/half_s1.pkl,ckpt/paperx/half_s2.pkl
echo ABLATION_GATES_MAIN_DONE
