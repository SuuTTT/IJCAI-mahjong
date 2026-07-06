#!/bin/bash
# E6 harness: 4-cell matrix cand_tau {0,2} x ref_tau {0,2}, x3 wall-seed blocks.
# Candidate = moyu_bn_128x40. Emits per-game single-game metrics + duplicate placement.
# CPU-only (GPUs reserved for E4). Free-guarded: honors /root/STOP_E6, watches disk.
cd /root/IJCAI-mahjong/train/caiest_repro || exit 1
LOG=/root/e6_eval.log; GD=ckpt/e6/gates; mkdir -p "$GD"
MOYU=/root/assets/moyu_bn_128x40.pkl
KIND=resbn; CFG=channels=128,blocks=40
SEEDS=300; W=48
SEED0S="70000 80000 90000"
echo "$(date -u) e6 START pid=$$" >> "$LOG"
for ct in 0 2; do
  for rt in 0 2; do
    for s0 in $SEED0S; do
      [ -f /root/STOP_E6 ] && { echo "$(date -u) STOP_E6 seen, exiting" >> "$LOG"; exit 0; }
      USE=$(df / | awk "NR==2{gsub(/%/,\"\",\$5); print \$5}")
      if [ "$USE" -ge 90 ]; then echo "$(date -u) DISK $USE% >=90, abort" >> "$LOG"; exit 1; fi
      gj="$GD/ct${ct}_rt${rt}_s${s0}.npz"
      [ -f "$gj" ] && { echo "$(date -u) skip $gj" >> "$LOG"; continue; }
      echo "$(date -u) RUN ct$ct rt$rt s$s0" >> "$LOG"
      python3 e6_gate.py --cand "$MOYU" --cand-kind "$KIND" --cand-cfg "$CFG" \
        --ref "$MOYU" --ref-kind "$KIND" --ref-cfg "$CFG" \
        --claim-tau "$ct" --ref-tau "$rt" --seeds "$SEEDS" --workers "$W" --seed0 "$s0" --out "$gj" \
        >> "$LOG" 2>&1 && echo "$(date -u) DONE ct$ct rt$rt s$s0" >> "$LOG" \
        || echo "$(date -u) FAIL ct$ct rt$rt s$s0" >> "$LOG"
    done
  done
done
echo "$(date -u) e6 ALL CELLS DONE -> aggregating" >> "$LOG"
python3 e6_aggregate.py >> "$LOG" 2>&1 && echo "$(date -u) e6 AGGREGATE DONE" >> "$LOG" || echo "$(date -u) AGG FAIL" >> "$LOG"
echo "$(date -u) e6 END" >> "$LOG"
