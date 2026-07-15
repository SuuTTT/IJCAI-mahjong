#!/bin/bash
cd /root/caiest_repro
A=ckpt/aug/aug_128x40_s0.pkl
EA=ckpt/paperx/djA_s0.pkl,ckpt/paperx/djA_s1.pkl,ckpt/paperx/djA_s2.pkl
EB=ckpt/paperx/djB_s0.pkl,ckpt/paperx/djB_s1.pkl,ckpt/paperx/djB_s2.pkl
EX=ckpt/paperx/djA_s0.pkl,ckpt/paperx/djA_s1.pkl,ckpt/paperx/djB_s0.pkl
mkdir -p kd_blocks
for i in $(seq 0 11); do
  [ -f kd_blocks/djAens_b$i.json ] || python3 e12_ens_gate.py --cand $EA --ref $A --seeds 500 --workers 100 --seed0 $((360000 + i*1000)) --out kd_blocks/djAens_b$i.json
  [ -f kd_blocks/djBens_b$i.json ] || python3 e12_ens_gate.py --cand $EB --ref $A --seeds 500 --workers 100 --seed0 $((360000 + i*1000)) --out kd_blocks/djBens_b$i.json
  [ -f kd_blocks/djXens_b$i.json ] || python3 e12_ens_gate.py --cand $EX --ref $A --seeds 500 --workers 100 --seed0 $((360000 + i*1000)) --out kd_blocks/djXens_b$i.json
done
echo DJ_GATES_DONE
