#!/bin/bash
# E2 harness: 8-cell matrix cand_tau {0,2} x ref_tau {0,1,2,3}, x3 wall-seed blocks for mean+/-std.
# Candidate = moyu_bn_128x40 (canonical). Free-guarded: honors /root/STOP_E2 and watches disk.
cd /root/IJCAI-mahjong/train/caiest_repro || exit 1
LOG=/root/e2_eval.log; GD=ckpt/e2/gates; mkdir -p "$GD"
MOYU=/root/assets/moyu_bn_128x40.pkl
KIND=resbn; CFG=channels=128,blocks=40
SEEDS=300; W=64
SEED0S="70000 80000 90000"
echo "$(date -u) e2 START pid=$$" >> "$LOG"
for ct in 0 2; do
  for rt in 0 1 2 3; do
    for s0 in $SEED0S; do
      [ -f /root/STOP_E2 ] && { echo "$(date -u) STOP_E2 seen, exiting" >> "$LOG"; exit 0; }
      # disk guard: bail if >92% full
      USE=$(df / | awk "NR==2{gsub(/%/,\"\",\$5); print \$5}")
      if [ "$USE" -ge 92 ]; then echo "$(date -u) DISK $USE% >=92, abort" >> "$LOG"; exit 1; fi
      gj="$GD/ct${ct}_rt${rt}_s${s0}.json"
      [ -f "$gj" ] && { echo "$(date -u) skip $gj" >> "$LOG"; continue; }
      echo "$(date -u) RUN ct$ct rt$rt s$s0" >> "$LOG"
      python3 e2_gate.py --cand "$MOYU" --cand-kind "$KIND" --cand-cfg "$CFG" \
        --ref "$MOYU" --ref-kind "$KIND" --ref-cfg "$CFG" \
        --claim-tau "$ct" --ref-tau "$rt" --seeds "$SEEDS" --workers "$W" --seed0 "$s0" --out "$gj" \
        >> "$LOG" 2>&1 && echo "$(date -u) DONE ct$ct rt$rt s$s0" >> "$LOG" \
        || echo "$(date -u) FAIL ct$ct rt$rt s$s0" >> "$LOG"
    done
  done
done
echo "$(date -u) e2 ALL CELLS DONE -> aggregating" >> "$LOG"
python3 e2_aggregate.py >> "$LOG" 2>&1 && echo "$(date -u) e2 AGGREGATE DONE" >> "$LOG" || echo "$(date -u) AGG FAIL" >> "$LOG"
echo "$(date -u) e2 END" >> "$LOG"
