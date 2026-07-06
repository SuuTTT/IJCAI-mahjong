#!/bin/bash
cd /root/IJCAI-mahjong/train/caiest_repro
K3=ckpt/kd/kd_128x40_s0.pkl,ckpt/kd/kd_128x40_s1.pkl,ckpt/kd/kd_128x40_s2.pkl
mkdir -p kd_blocks
for i in $(seq 0 5); do
  while [ "$(awk '{print ($1>105)?1:0}' /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f kd_blocks/shd1_b$i.json ] || python3 e16b_shanten_gate.py --cand $K3 --ref $K3 --sh-push 1 --min-turn 6 --min-melds 1 --top-k 8 --seeds 500 --workers 60 --seed0 $((620000 + i*1000)) --out kd_blocks/shd1_b$i.json
done
echo MAIN_BLOCKS_DONE
