#!/bin/bash
# E1 over-claiming study — from-scratch BC training sweep. Free-guarded, 1 job/GPU, STOP_E1 flag,
# disk-watched. Mirrors kbclaim.sh. Trains the capacity / composition / data-fraction matrix.
cd /root/IJCAI-mahjong/train/caiest_repro || exit 1
LOG=/root/e1_train.log; mkdir -p ckpt/e1 ckpt/e1/meas ckpt/e1/gates
STEPS=${STEPS:-24000}
END=$(( $(date +%s) + 36000 ))   # 10h cap
echo "$(date -u) e1_train START steps=$STEPS" >> "$LOG"
gpu_free(){ local m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null|tr -d ' '); [ -n "$m" ]&&[ "$m" -lt 800 ]; }
disk_ok(){ local u=$(df -P /root|awk 'NR==2{print $5}'|tr -d '%'); [ "$u" -lt 92 ]; }

# label channels blocks data frac seed  (data: "full" or an npz path)
CFGS=(
"full_64x6_s0    64  6  full                              1.0 0"
"full_64x6_s1    64  6  full                              1.0 1"
"full_128x20_s0  128 20 full                              1.0 0"
"full_128x20_s1  128 20 full                              1.0 1"
"full_128x40_s0  128 40 full                              1.0 0"
"full_128x40_s1  128 40 full                              1.0 1"
"full_256x40_s0  256 40 full                              1.0 0"
"full_256x40_s1  256 40 full                              1.0 1"
"top_128x40_s0   128 40 data/teachers/toponly_disjoint.npz 1.0 0"
"top_128x40_s1   128 40 data/teachers/toponly_disjoint.npz 1.0 1"
"frac25_128x40_s0 128 40 full                             0.25 0"
"frac50_128x40_s0 128 40 full                             0.50 0"
)
declare -A SLOT; declare -A SLBL; qi=0
while [ "$(date +%s)" -lt "$END" ] && [ ! -f /root/STOP_E1 ]; do
  for g in 0 1 2 3; do
    pid=${SLOT[$g]:-}
    if { [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; } && gpu_free "$g" && disk_ok; then
      [ "$qi" -ge "${#CFGS[@]}" ] && continue
      set -- ${CFGS[$qi]}; lbl=$1; ch=$2; blk=$3; data=$4; frac=$5; seed=$6
      out="ckpt/e1/${lbl}.pkl"
      if [ ! -f "$out" ]; then
        CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 nohup python3 e1_train.py \
          --channels "$ch" --blocks "$blk" --data "$data" --frac "$frac" --seed "$seed" \
          --steps "$STEPS" --out "$out" >> "$LOG" 2>&1 &
        SLOT[$g]=$!; SLBL[$g]=$lbl; echo "$(date -u) GPU$g START $lbl (ch$ch blk$blk frac$frac seed$seed steps$STEPS)" >> "$LOG"
        qi=$(( qi + 1 )); sleep 25
      else
        echo "$(date -u) SKIP $lbl (exists)" >> "$LOG"; qi=$(( qi + 1 ))
      fi
    fi
  done
  if [ "$qi" -ge "${#CFGS[@]}" ]; then
    busy=0; for g in 0 1 2 3; do kill -0 "${SLOT[$g]:-0}" 2>/dev/null && busy=1; done
    [ "$busy" -eq 0 ] && break
  fi
  sleep 60
done
echo "$(date -u) e1_train END qi=$qi" >> "$LOG"
