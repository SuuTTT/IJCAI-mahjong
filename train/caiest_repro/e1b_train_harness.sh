#!/bin/bash
# E1b FULL-CONVERGENCE over-claiming study. 90000 steps (~16-epoch official budget on the 5.87M
# E1 train split) to reach ~0.894 val. Capacity push to 384x40. Free-guarded, 1 job/GPU,
# STOP_E1B flag, disk-watched. Same e1_train.py recipe as E1 (only --steps differs).
cd /root/IJCAI-mahjong/train/caiest_repro || exit 1
LOG=/root/e1b_train.log; mkdir -p ckpt/e1b ckpt/e1b/meas ckpt/e1b/gates
STEPS=${STEPS:-90000}
END=$(( $(date +%s) + 86400 ))   # 24h cap
echo "$(date -u) e1b_train START steps=$STEPS" >> "$LOG"
gpu_free(){ local m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null|tr -d " "); [ -n "$m" ]&&[ "$m" -lt 800 ]; }
disk_ok(){ local u=$(df -P /root|awk "NR==2{print \$5}"|tr -d "%"); [ "$u" -lt 90 ]; }

# label channels blocks seed
CFGS=(
"full_128x40_s0  128 40 0"
"full_128x40_s1  128 40 1"
"full_256x40_s0  256 40 0"
"full_256x40_s1  256 40 1"
"full_384x40_s0  384 40 0"
)
declare -A SLOT; declare -A SLBL; qi=0
while [ "$(date +%s)" -lt "$END" ] && [ ! -f /root/STOP_E1B ]; do
  for g in 0 1 2 3; do
    pid=${SLOT[$g]:-}
    if { [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; } && gpu_free "$g" && disk_ok; then
      [ "$qi" -ge "${#CFGS[@]}" ] && continue
      set -- ${CFGS[$qi]}; lbl=$1; ch=$2; blk=$3; seed=$4
      out="ckpt/e1b/${lbl}.pkl"
      if [ ! -f "$out" ]; then
        CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 nohup python3 e1_train.py \
          --channels "$ch" --blocks "$blk" --data full --frac 1.0 --seed "$seed" \
          --steps "$STEPS" --out "$out" >> "$LOG" 2>&1 &
        SLOT[$g]=$!; SLBL[$g]=$lbl; echo "$(date -u) GPU$g START $lbl (ch$ch blk$blk seed$seed steps$STEPS)" >> "$LOG"
        qi=$(( qi + 1 )); sleep 30
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
echo "$(date -u) e1b_train END qi=$qi" >> "$LOG"
