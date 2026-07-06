#!/bin/bash
# run_bestnet_train.sh — ENHANCED-recipe (e11) training of the BIG candidate nets, 1 job/GPU,
# free-guard + disk-guard(<2200MB no-start / <2000MB abort) + /root/STOP_BESTNET. setsid.
# Deletes each .bn.pkl once its fused .pkl exists (disk hygiene). Same e11 recipe as aug_s0.
cd /root/IJCAI-mahjong/train/caiest_repro || exit 1
LOG=/root/bestnet_train.log; mkdir -p ckpt/best
# gpu ch blk seed steps   (384x40 ~0.5s/step -> 60k~8.3h ; 192x40 ~0.19s/step -> 130k~7h)
CFGS=("0 384 40 0 60000" "1 384 40 1 60000" "2 192 40 0 130000" "3 192 40 1 130000")
END=$(( $(date +%s) + 54000 ))   # 15h hard cap
echo "$(date -u) run_bestnet_train START" >> "$LOG"
gpu_free(){ local m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null|tr -d " "); [ -n "$m" ] && [ "$m" -lt 15000 ]; }
disk_mb(){ df -m /root|awk "NR==2{print \$4}"; }
declare -A SLOT
while [ "$(date +%s)" -lt "$END" ] && [ ! -f /root/STOP_BESTNET ]; do
  alldone=1
  # disk hygiene: drop .bn.pkl whose fused .pkl exists
  for f in ckpt/best/enh_*.pkl; do
    [ -f "$f" ] || continue; case "$f" in *.bn.pkl) continue;; esac
    bn="${f%.pkl}.bn.pkl"; [ -f "$bn" ] && rm -f "$bn" && echo "$(date -u) rmbn $bn" >> "$LOG"
  done
  [ "$(disk_mb)" -lt 2000 ] && { echo "$(date -u) DISK<2000MB ABORT" >> "$LOG"; break; }
  for cfg in "${CFGS[@]}"; do
    set -- $cfg; g=$1; ch=$2; blk=$3; seed=$4; steps=$5
    out="ckpt/best/enh_${ch}x${blk}_s${seed}.pkl"
    [ -f "$out" ] && continue
    alldone=0
    key="${ch}_${seed}"; pid=${SLOT[$key]:-}
    if { [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; }; then
      if gpu_free "$g" && [ "$(disk_mb)" -ge 2200 ]; then
        CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=6 setsid python3 e11_train.py \
          --channels "$ch" --blocks "$blk" --seed "$seed" --steps "$steps" --out "$out" >> "$LOG" 2>&1 < /dev/null &
        SLOT[$key]=$!
        echo "$(date -u) GPU$g START ${ch}x${blk}_s${seed} steps=$steps pid=${SLOT[$key]}" >> "$LOG"
        sleep 45
      fi
    fi
  done
  [ "$alldone" -eq 1 ] && break
  sleep 60
done
# final bn cleanup
for f in ckpt/best/enh_*.pkl; do case "$f" in *.bn.pkl) rm -f "$f";; esac; done
echo "$(date -u) run_bestnet_train END" >> "$LOG"
