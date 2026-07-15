#!/bin/bash
# Keep-warm: heterogeneous teacher bank (diverse data-views x objectives) for the SE graph/diversity
# experiments. CAPPED at 1 job/GPU (avoid the load-122 stall). Coexists; grabs only genuinely-free GPUs.
cd /root/caiest_repro
T6=ckpt/aug/aug_128x40_s0.pkl,ckpt/aug/aug_128x40_s1.pkl,ckpt/aug/aug_128x40_s2.pkl,ckpt/aug/aug_128x40_s3.pkl,ckpt/aug/aug_128x40_s4.pkl,ckpt/aug/aug_128x40_s5.pkl
mkdir -p ckpt/hetbank logs/hetbank
free_gpu(){ for g in 0 1 2 3 4 5 6 7; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null|grep -c .)
  [ "${m:-99999}" -lt 4000 ] && [ "${n:-9}" -lt 1 ] && { echo $g; return; }   # 1 job/GPU cap
done; echo -1; }
seed=8500
while [ ! -f /root/STOP_HETBANK ]; do
  for frac in 0.30 0.50 0.70 1.00; do for alpha in 0.5 0.7 1.0; do
    [ -f /root/STOP_HETBANK ] && break 2
    ftag=$(echo $frac|sed 's/\.//'); atag=$(echo $alpha|sed 's/\.//')
    out=ckpt/hetbank/het_f${ftag}_a${atag}_s${seed}.pkl
    [ -f "$out" ] && { seed=$((seed+1)); continue; }
    g=-1; while [ "$g" = "-1" ]; do [ -f /root/STOP_HETBANK ] && break 3; g=$(free_gpu); [ "$g" = "-1" ] && sleep 45; done
    CUDA_VISIBLE_DEVICES=$g nohup python3 e13_kd_frac.py --channels 128 --blocks 40 --steps 40000 --seed $seed --teachers $T6 --alpha $alpha --frac $frac --out $out > logs/hetbank/het_${ftag}_${atag}_${seed}.log 2>&1 &
    echo "$(date -u +%H:%M) het frac=$frac alpha=$alpha seed=$seed gpu=$g" >> logs/hetbank/bank.log
    seed=$((seed+1)); sleep 30
  done; done
done
