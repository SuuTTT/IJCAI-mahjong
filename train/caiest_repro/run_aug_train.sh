#!/bin/bash
# run_aug_train.sh — ENHANCED-recipe 128x40 training, 3 seeds, one per free GPU (0,1,2).
# GPU3 left to the neighbor; free-guard (skip busy GPUs), disk guard, /root/STOP_AUG honored.
cd /root/IJCAI-mahjong/train/caiest_repro || exit 1
LOG=/root/aug_train.log; mkdir -p ckpt/aug
STEPS=${STEPS:-130000}
END=$(( $(date +%s) + 43200 ))   # 12h cap
echo "$(date -u) run_aug_train START steps=$STEPS" >> "$LOG"
gpu_free(){ local m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null|tr -d ' '); [ -n "$m" ] && [ "$m" -lt 15000 ]; }
disk_ok(){ local u=$(df -P /root|awk 'NR==2{print $5}'|tr -d '%'); [ "$u" -lt 92 ]; }

# gpu seed
CFGS=("0 0" "1 1" "2 2")
declare -A SLOT
while [ "$(date +%s)" -lt "$END" ] && [ ! -f /root/STOP_AUG ]; do
  alldone=1
  for cfg in "${CFGS[@]}"; do
    set -- $cfg; g=$1; seed=$2
    out="ckpt/aug/aug_128x40_s${seed}.pkl"
    [ -f "$out" ] && continue
    alldone=0
    pid=${SLOT[$seed]:-}
    if { [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; }; then
      if gpu_free "$g" && disk_ok; then
        CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=6 nohup python3 e11_train.py \
          --seed "$seed" --steps "$STEPS" --out "$out" >> "$LOG" 2>&1 &
        SLOT[$seed]=$!
        echo "$(date -u) GPU$g START seed$seed pid=${SLOT[$seed]}" >> "$LOG"
        sleep 40
      fi
    fi
  done
  [ "$alldone" -eq 1 ] && break
  sleep 60
done
echo "$(date -u) run_aug_train END" >> "$LOG"
