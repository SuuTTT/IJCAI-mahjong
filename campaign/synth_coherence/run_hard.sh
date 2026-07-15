#!/bin/bash
set -u
cd /root/synth_coherence
OUT=/root/synth_coherence/cells_hard
LOGS=/root/synth_coherence/logs_hard
mkdir -p "$OUT" "$LOGS"
# harder regime: K=20 classes, higher freq features (tune difficulty toward clean acc ~0.65-0.7)
export K=20
SEEDS="0,1,2,3,4,5,6,7"
CS=(1 2 4 8)
EPS=(0.0 0.1 0.2 0.3 0.4)
# difficulty check first (logged)
python3 synth_hard.py --mode calibrate --gpu 0 > "$LOGS/calibrate.log" 2>&1
cells=()
for c in "${CS[@]}"; do for e in "${EPS[@]}"; do cells+=("$c:$e"); done; done
worker() {
  local gpu=$1 idx=0
  for cell in "${cells[@]}"; do
    if [ $((idx % 4)) -eq "$gpu" ]; then
      local C=${cell%%:*} E=${cell##*:}
      local ff="$OUT/cell_C${C}_eps$(printf %.2f "$E").json"
      if [ ! -f "$ff" ]; then
        python3 synth_hard.py --mode cell --C "$C" --eps "$E" --seeds "$SEEDS" --gpu "$gpu" --outdir "$OUT" > "$LOGS/cell_C${C}_eps${E}.log" 2>&1
      fi
    fi
    idx=$((idx+1))
  done
}
for g in 0 1 2 3; do worker $g & done
wait
touch /root/synth_coherence/SYNTH_HARD_GRID_DONE
echo GRID_DONE
