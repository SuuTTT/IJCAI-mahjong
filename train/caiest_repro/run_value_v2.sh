#!/bin/bash
# run_value_v2.sh — VALUE-HEAD V2 pipeline: cook labels -> verify -> 6 training jobs on
# free GPUs (memory-picked from {2,3,4,6,7}, max 4) -> aggregate results/VALUE_V2.json.
# Durable: run under setsid; single flock-guarded job queue (no GPU double-booking).
set -u
cd /root/caiest_repro
mkdir -p logs results
RUN=results/VALUE_V2_RUNNING; FAIL=results/VALUE_V2_FAILED; date > "$RUN"; rm -f "$FAIL"

echo "[$(date)] === VALUE V2 pipeline start ==="

# 1) cook official value labels (skip if already verified)
if python3 -c "
import numpy as np,sys
try: d=np.load('data/official_value_labels.npz'); sys.exit(0 if int(d['verified'])==1 else 1)
except Exception: sys.exit(1)"; then
  echo "[$(date)] labels already cooked+verified, skipping"
else
  echo "[$(date)] cooking official value labels (96 workers)"
  python3 cook_value_labels.py --workers 96 > logs/cook_value_labels.log 2>&1
  rc=$?
  if [ $rc -ne 0 ] || ! grep -q ALIGN_OK logs/cook_value_labels.log; then
    echo "[$(date)] COOK/ALIGN FAILED rc=$rc — aborting"; cp logs/cook_value_labels.log "$FAIL"; rm -f "$RUN"; exit 2
  fi
  echo "[$(date)] labels cooked + ALIGN_OK"
fi

# 2) pick free GPUs from the allowed pool (CIFAR on 0-1, RL pilot on 5 — do not touch)
GPUS=""
NG=0
for g in 2 3 4 6 7; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  if [ "$used" -lt 500 ] && [ $NG -lt 4 ]; then GPUS="$GPUS $g"; NG=$((NG+1)); fi
done
if [ $NG -lt 1 ]; then echo "no free GPUs" | tee "$FAIL"; rm -f "$RUN"; exit 3; fi
echo "[$(date)] using GPUs:$GPUS"

# 3) job queue (flock-guarded pop)
Q=results/value_v2_jobs.list
printf "%s\n" a:0 a:1 b:0 b:1 c:0 c:1 > "$Q"
pop_job() {
  (
    flock -x 200
    j=$(head -n1 "$Q")
    [ -n "$j" ] && sed -i '1d' "$Q"
    echo "$j"
  ) 200>"$Q.lock"
}
runner() {
  local gpu=$1
  while :; do
    j=$(pop_job); [ -z "$j" ] && break
    v=${j%:*}; s=${j#*:}
    out=results/value_v2_${v}_s${s}.json
    if [ -f "$out" ]; then echo "[$(date)] gpu$gpu skip $j (exists)"; continue; fi
    echo "[$(date)] gpu$gpu start $j"
    CUDA_VISIBLE_DEVICES=$gpu python3 f2_value_v2.py --variant "$v" --seed "$s" \
      --out "$out" > logs/value_v2_${v}_s${s}.log 2>&1
    echo "[$(date)] gpu$gpu done $j rc=$? ($(ls -la $out 2>/dev/null | awk '{print $5}') bytes)"
  done
}
for g in $GPUS; do runner "$g" & done
wait
echo "[$(date)] all training jobs finished"

# 4) aggregate
n=$(ls results/value_v2_[abc]_s[01].json 2>/dev/null | wc -l)
if [ "$n" -lt 6 ]; then
  echo "[$(date)] only $n/6 job jsons present — aggregating anyway, marking FAILED"
  echo "only $n/6 jobs produced output" > "$FAIL"
fi
python3 agg_value_v2.py > logs/agg_value_v2.log 2>&1 && cat logs/agg_value_v2.log
rm -f "$RUN"; date > results/VALUE_V2_DONE
echo "[$(date)] === VALUE V2 pipeline DONE (results/VALUE_V2.json) ==="
