#!/bin/bash
# recipe_gate.sh — RECIPE sweep GATE loop. Calibrates (aug_s0 vs aug_s0 must=2.500), then gates
# every finished recipe net vs aug_s0 to FULL blocks (calibrated e11_gate lam=0). CPU workers only
# (good neighbour). Re-aggregates -> RECIPE_RESULTS.json + RECIPE_WRITEUP.md after each config.
# /root/STOP_RECIPE honored, disk-guard.
cd /root/IJCAI-mahjong/train/caiest_repro || exit 1
LOG=/root/recipe_gate.log; mkdir -p ckpt/recipe/gates
REF=ckpt/aug/aug_128x40_s0.pkl
SEEDS=${SEEDS:-500}; WORKERS=${WORKERS:-40}; FULL=${FULL:-12}
END=$(( $(date +%s) + 43200 ))   # 12h cap
echo "$(date -u) recipe_gate START seeds=$SEEDS workers=$WORKERS blocks=$FULL ref=$REF" >> "$LOG"
disk_mb(){ df -m /root|awk 'NR==2{print $4}'; }

do_gate(){  # tag  cand  [extra tta args...]
  local tag=$1; local cand=$2; shift 2
  for b in $(seq 0 $((FULL-1))); do
    [ -f /root/STOP_RECIPE ] && { echo "$(date -u) STOP halt" >> "$LOG"; return; }
    [ "$(disk_mb)" -lt 2000 ] && { echo "$(date -u) DISK LOW halt" >> "$LOG"; return; }
    local s0=$(( 500000 + b*1000 ))
    local out="ckpt/recipe/gates/${tag}_s${s0}.json"
    [ -f "$out" ] && continue
    python3 e11_gate.py --cand "$cand" --ref "$REF" --seeds "$SEEDS" --workers "$WORKERS" \
        --seed0 "$s0" "$@" --out "$out" >> "$LOG" 2>&1
    echo "$(date -u) gated $tag block $b rc=$?" >> "$LOG"
  done
}

# calibration (1 block): aug_s0 vs aug_s0 must be 2.500
if [ ! -f ckpt/recipe/gates/calib_s500000.json ]; then
  python3 e11_gate.py --cand "$REF" --ref "$REF" --seeds "$SEEDS" --workers "$WORKERS" \
      --seed0 500000 --out ckpt/recipe/gates/calib_s500000.json >> "$LOG" 2>&1
  echo "$(date -u) calib done" >> "$LOG"
fi

while [ "$(date +%s)" -lt "$END" ] && [ ! -f /root/STOP_RECIPE ]; do
  progressed=0
  for cand in ckpt/recipe/*.pkl; do
    [ -f "$cand" ] || continue
    case "$cand" in *.bn.pkl) continue;; esac
    tag=$(basename "$cand" .pkl)
    last="ckpt/recipe/gates/${tag}_s$(( 500000 + (FULL-1)*1000 )).json"
    [ -f "$last" ] && continue           # already fully gated
    echo "$(date -u) gating $tag ($cand)" >> "$LOG"
    do_gate "$tag" "$cand"
    python3 recipe_agg.py >> "$LOG" 2>&1
    progressed=1
  done
  # exit if training harness is gone AND everything present is fully gated
  if [ "$progressed" -eq 0 ]; then
    if ! pgrep -f "recipe_train.sh" >/dev/null 2>&1 && ! pgrep -f "e11_train.py.*ckpt/recipe/" >/dev/null 2>&1; then
      echo "$(date -u) all gated + trainer gone, exit" >> "$LOG"; break
    fi
    sleep 120
  fi
done
python3 recipe_agg.py >> "$LOG" 2>&1
echo "$(date -u) recipe_gate END" >> "$LOG"
