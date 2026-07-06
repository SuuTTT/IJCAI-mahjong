#!/bin/bash
# E3 value-model capacity-scaling harness. Free-guarded, 1 job/GPU, honors /root/STOP_E3.
# Usage: e3_harness.sh <gpu> <job1> [job2 ...]   where job = "CHxBL:SEED"
set -u
cd /root/IJCAI-mahjong/train/caiest_repro
GPU=$1; shift
LOGDIR=logs/e3; mkdir -p "$LOGDIR" ckpt/e3 E3_json
HLOG="$LOGDIR/gpu${GPU}.harness.log"
echo "[$(date)] harness start gpu=$GPU jobs=$*" >>"$HLOG"

free_guard() {
  # skip/wait if this GPU has >800MB used by someone else
  for _ in $(seq 1 720); do  # up to ~6h of 30s waits
    [ -f /root/STOP_E3 ] && { echo "[$(date)] STOP_E3 present, aborting" >>"$HLOG"; exit 0; }
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    if [ "${used:-9999}" -le 800 ]; then return 0; fi
    echo "[$(date)] gpu$GPU busy (${used}MB>800), wait 30s" >>"$HLOG"; sleep 30
  done
  echo "[$(date)] gpu$GPU never freed, giving up" >>"$HLOG"; return 1
}

disk_guard() {
  pct=$(df --output=pcent / | tail -1 | tr -dc '0-9')
  if [ "${pct:-0}" -ge 92 ]; then
    echo "[$(date)] disk ${pct}% >=92, cleaning tmp" >>"$HLOG"
    rm -f /tmp/*.npz /tmp/*.npy 2>/dev/null
  fi
}

for JOB in "$@"; do
  [ -f /root/STOP_E3 ] && { echo "[$(date)] STOP_E3, stop queue" >>"$HLOG"; break; }
  CB="${JOB%%:*}"; SEED="${JOB##*:}"
  CH="${CB%x*}"; BL="${CB#*x}"
  TAG="${CH}x${BL}_s${SEED}"
  OUT="ckpt/e3/value_${TAG}.pkl"
  JOUT="E3_json/${TAG}.json"
  RLOG="$LOGDIR/${TAG}.log"
  if [ -f "$JOUT" ]; then echo "[$(date)] $TAG already has json, skip" >>"$HLOG"; continue; fi
  disk_guard
  echo "[$(date)] free_guard gpu$GPU for $TAG" >>"$HLOG"
  free_guard || continue
  echo "[$(date)] START $TAG ch=$CH bl=$BL seed=$SEED gpu=$GPU" >>"$HLOG"
  python3 train_value.py --gpu "$GPU" --channels "$CH" --blocks "$BL" \
      --epochs 8 --seed "$SEED" --out "$OUT" --json_out "$JOUT" >"$RLOG" 2>&1
  rc=$?
  echo "[$(date)] DONE $TAG rc=$rc" >>"$HLOG"
  disk_guard
done
echo "[$(date)] harness gpu$GPU queue complete" >>"$HLOG"
