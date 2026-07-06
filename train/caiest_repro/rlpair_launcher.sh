#!/bin/bash
# wait for ensemble gate to finish (frees 64 CPU cores), then start paired-wall RL on GPU3
cd /root/IJCAI-mahjong/train/caiest_repro
while pgrep -f e12_ens_gate > /dev/null || [ "$(awk "{print (\$1>70)?1:0}" /proc/loadavg)" = "1" ]; do sleep 120; done
rm -f /root/STOP_RL
CUDA_VISIBLE_DEVICES=3 python3 rl_paired.py --actors 16 --games-per-actor 2 --minutes 900 \
  --snap-every 25 --tag rlpair > logs/rl_paired.log 2>&1
echo "RL_PAIRED DONE" >> logs/rl_paired.log
