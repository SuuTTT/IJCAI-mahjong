#!/bin/bash
# Keep-warm daemon: self-refilling KD sample-efficiency sweep (data-fraction x seed).
# Coexists with the SE agent via a memory-based GPU picker; runs ~8-12h until STOP file.
cd /root/caiest_repro
mkdir -p ckpt/sampeff logs/sampeff
T6=ckpt/aug/aug_128x40_s0.pkl,ckpt/aug/aug_128x40_s1.pkl,ckpt/aug/aug_128x40_s2.pkl,ckpt/aug/aug_128x40_s3.pkl,ckpt/aug/aug_128x40_s4.pkl,ckpt/aug/aug_128x40_s5.pkl
FRACS="0.02 0.03 0.05 0.07 0.10 0.15 0.25 0.50 1.00"
free_gpu() {
  for g in 0 1 2 3 4 5 6 7; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null | grep -c .)
    [ "${m:-99999}" -lt 3000 ] && [ "${n:-9}" -lt 2 ] && { echo $g; return; }
  done
  echo -1
}
seed=8000
round=0
while [ ! -f /root/STOP_SAMPEFF ]; do
  for frac in $FRACS; do
    [ -f /root/STOP_SAMPEFF ] && break
    ftag=$(echo $frac | sed 's/\.//')
    out=ckpt/sampeff/kdf_f${ftag}_s${seed}.pkl
    [ -f "$out" ] && { seed=$((seed+1)); continue; }
    g=-1
    while [ "$g" = "-1" ]; do
      [ -f /root/STOP_SAMPEFF ] && break 2
      g=$(free_gpu); [ "$g" = "-1" ] && sleep 45
    done
    CUDA_VISIBLE_DEVICES=$g nohup python3 e13_kd_frac.py --channels 128 --blocks 40 --steps 25000 --seed $seed --teachers $T6 --alpha 0.7 --frac $frac --out $out > logs/sampeff/kdf_f${ftag}_s${seed}.log 2>&1 &
    echo "$(date -u +%H:%M) launched frac=$frac seed=$seed gpu=$g" >> logs/sampeff/daemon.log
    seed=$((seed+1)); sleep 25
  done
  round=$((round+1))
  echo "$(date -u +%H:%M) completed round $round, seed now $seed" >> logs/sampeff/daemon.log
done
echo "$(date -u +%H:%M) STOP file seen, daemon exiting" >> logs/sampeff/daemon.log
