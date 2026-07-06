#!/bin/bash
# run_pimc.sh — self-managing multi-block PIMC search gate over 3 compute tiers.
# Honors /root/STOP_SEARCH, guards disk free, moderate CPU workers (GPU training untouched).
set -u
cd /root/IJCAI-mahjong/train/caiest_repro
CAND=ckpt/aug/aug_128x40_s0.pkl
REF=ckpt/aug/aug_128x40_s0.pkl
VAL=ckpt/value_256x40.pkl
JDIR=pimc_json
mkdir -p "$JDIR"
WORKERS=48
SEEDS=300               # 1200 games/block
BLOCKS=10
LOG=pimc_harness.log

# tiers: "N H tag"  (A ~500ms, B ~2000ms, C ~4000ms)
TIERS=("10 12 A" "20 20 B" "40 30 C")

echo "[$(date -u +%H:%M:%S)] PIMC harness start: ${#TIERS[@]} tiers x $BLOCKS blocks x $SEEDS seeds, $WORKERS workers" >> "$LOG"

for tier in "${TIERS[@]}"; do
  set -- $tier; N=$1; H=$2; TAG=$3
  for b in $(seq 0 $((BLOCKS-1))); do
    OUT="$JDIR/N${N}_H${H}_b${b}.json"
    [ -s "$OUT" ] && { echo "[$(date -u +%H:%M:%S)] skip existing $OUT" >> "$LOG"; continue; }
    if [ -f /root/STOP_SEARCH ]; then echo "[$(date -u +%H:%M:%S)] STOP_SEARCH -> abort" >> "$LOG"; exit 0; fi
    FREE=$(df -m --output=avail / | tail -1 | tr -d ' ')
    if [ "$FREE" -lt 1500 ]; then echo "[$(date -u +%H:%M:%S)] LOW DISK ${FREE}MB -> abort" >> "$LOG"; exit 0; fi
    SEED0=$((900000 + N*10000 + H*100 + b*1000))
    echo "[$(date -u +%H:%M:%S)] tier $TAG N=$N H=$H block $b seed0=$SEED0 free=${FREE}MB" >> "$LOG"
    python3 pimc_gate.py --cand "$CAND" --ref "$REF" --value "$VAL" \
       --N "$N" --H "$H" --topk 5 --delta 3.0 --margin 0.0 \
       --seeds "$SEEDS" --workers "$WORKERS" --seed0 "$SEED0" --out "$OUT" \
       >> "$LOG" 2>&1
    echo "[$(date -u +%H:%M:%S)] done block $b: $(python3 -c "import json;d=json.load(open('$OUT'));print('pl=%.4f ms=%.0f ovr=%.3f'%(d['placement_pts'],d['per_move_ms'],d['override_rate']))" 2>/dev/null)" >> "$LOG"
  done
  python3 pimc_aggregate.py "$JDIR" SEARCH_RESULTS.json >> "$LOG" 2>&1
done
echo "[$(date -u +%H:%M:%S)] PIMC harness COMPLETE" >> "$LOG"
python3 pimc_aggregate.py "$JDIR" SEARCH_RESULTS.json >> "$LOG" 2>&1
