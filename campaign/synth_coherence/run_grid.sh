#!/bin/bash
# Grid driver: C x eps, parallel across GPUs 0-3, one serial worker per GPU.
# Loud, DONE markers, [ -f ] guards for idempotent resume.
set -u
cd /root/synth_coherence
OUT=/root/synth_coherence/cells
LOGS=/root/synth_coherence/logs
mkdir -p "$OUT" "$LOGS"
export H_SUB=16
SEEDS="0,1,2,3,4,5,6,7"
CS=(1 2 4 8)
EPS=(0.0 0.1 0.2 0.3 0.4)

cells=()
for c in "${CS[@]}"; do for e in "${EPS[@]}"; do cells+=("$c:$e"); done; done

worker() {
  local gpu=$1 idx=0
  for cell in "${cells[@]}"; do
    if [ $((idx % 4)) -eq "$gpu" ]; then
      local C=${cell%%:*} E=${cell##*:}
      local ff; ff="$OUT/cell_C${C}_eps$(printf '%.2f' "$E").json"
      if [ -f "$ff" ]; then
        echo "SKIP gpu$gpu C=$C eps=$E (exists)"
      else
        echo "START gpu$gpu C=$C eps=$E $(date +%H:%M:%S)"
        H_SUB=16 python3 synth.py --mode cell --C "$C" --eps "$E" --seeds "$SEEDS" \
          --gpu "$gpu" --outdir "$OUT" > "$LOGS/cell_C${C}_eps${E}.log" 2>&1
        echo "DONE  gpu$gpu C=$C eps=$E rc=$? $(date +%H:%M:%S)"
      fi
    fi
    idx=$((idx+1))
  done
  echo "WORKER $gpu FINISHED"
}

echo "GRID_START $(date)"
for g in 0 1 2 3; do worker "$g" & done
wait
echo "ALL_CELLS_DONE $(date)"
python3 aggregate.py && echo "AGG_DONE" || echo "AGG_FAILED"
touch /root/synth_coherence/GRID_DONE
echo "GRID_DONE_MARKER_WRITTEN $(date)"
