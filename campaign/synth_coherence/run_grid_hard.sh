#!/bin/bash
# HARD-regime grid: C x eps + finer C=1 slice. Parallel across GPUs 0-3.
set -u
cd /root/synth_coherence
OUT=/root/synth_coherence/cells_hard
LOGS=/root/synth_coherence/logs
mkdir -p "$OUT" "$LOGS" /root/synth_coherence/results
export D=64 K=20 FREQ=5 H_SUB=48
SEEDS="0,1,2,3,4,5,6,7"
CS=(1 2 4 8)
EPS=(0.0 0.1 0.2 0.3 0.4)

cells=()
for c in "${CS[@]}"; do for e in "${EPS[@]}"; do cells+=("$c:$e"); done; done
# finer threshold slice at C=1: extra eps not already in the grid
for e in 0.05 0.15 0.5; do cells+=("1:$e"); done

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
        D=64 K=20 FREQ=5 H_SUB=48 python3 synth_hard.py --mode cell --C "$C" --eps "$E" \
          --seeds "$SEEDS" --gpu "$gpu" --outdir "$OUT" > "$LOGS/hard_C${C}_eps${E}.log" 2>&1
        echo "DONE  gpu$gpu C=$C eps=$E rc=$? $(date +%H:%M:%S)"
      fi
    fi
    idx=$((idx+1))
  done
  echo "WORKER $gpu FINISHED"
}

echo "HARD_GRID_START $(date)"
for g in 0 1 2 3; do worker "$g" & done
wait
echo "HARD_ALL_CELLS_DONE $(date)"
python3 aggregate_hard.py && echo "HARD_AGG_DONE" || echo "HARD_AGG_FAILED"
touch /root/synth_coherence/HARD_GRID_DONE
echo "HARD_GRID_DONE_MARKER $(date)"
