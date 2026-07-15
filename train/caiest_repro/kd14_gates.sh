#!/bin/bash
cd /root/caiest_repro
until [ -f ckpt/kd14/kd14_s0.pkl ] && [ -f ckpt/kd14/kd14_s1.pkl ] && [ -f ckpt/kd14/kd14_s2.pkl ]; do sleep 300; done
sleep 30
A=ckpt/aug/aug_128x40_s0.pkl
K14=ckpt/kd14/kd14_s0.pkl,ckpt/kd14/kd14_s1.pkl,ckpt/kd14/kd14_s2.pkl
mkdir -p kd_blocks
for i in $(seq 0 23); do
  [ -f kd_blocks/kd14ens_b$i.json ] || python3 e12_ens_gate.py --cand $K14 --ref $A --seeds 500 --workers 110 --seed0 $((360000 + i*1000)) --out kd_blocks/kd14ens_b$i.json
done
echo KD14_GATES_DONE
