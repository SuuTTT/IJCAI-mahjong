#!/bin/bash
# run_aug_gate.sh — GATE phase: calibrated duplicate placement gate of every enhanced net + TTA
# configs vs bn128s1. Waits for the 3 aug nets to finish training, then runs multi-block gates,
# aggregates -> AUG_RESULTS.json + AUG_WRITEUP.md. Good-neighbor CPU workers; /root/STOP_AUG honored.
cd /root/IJCAI-mahjong/train/caiest_repro || exit 1
LOG=/root/aug_gate.log; mkdir -p ckpt/aug/gates
BN=ckpt/e1b/full_128x40_s1.pkl        # bn128s1 (current best / reference)
SEEDS=${SEEDS:-500}
WORKERS=${WORKERS:-48}
BLOCKS=${BLOCKS:-14}                   # blocks per candidate (disjoint seed0)
echo "$(date -u) run_aug_gate START seeds=$SEEDS workers=$WORKERS blocks=$BLOCKS" >> "$LOG"

disk_ok(){ local u=$(df -P /root|awk 'NR==2{print $5}'|tr -d '%'); [ "$u" -lt 93 ]; }
gate(){  # tag  cand  extra-args...
  local tag=$1; local cand=$2; shift 2
  for b in $(seq 0 $((BLOCKS-1))); do
    [ -f /root/STOP_AUG ] && { echo "$(date -u) STOP_AUG halt" >> "$LOG"; return; }
    disk_ok || { echo "$(date -u) DISK LOW halt" >> "$LOG"; return; }
    local s0=$(( 500000 + b*1000 ))
    local out="ckpt/aug/gates/${tag}_s${s0}.json"
    [ -f "$out" ] && continue
    python3 e11_gate.py --cand "$cand" --ref "$BN" --seeds "$SEEDS" --workers "$WORKERS" \
        --seed0 "$s0" "$@" --out "$out" >> "$LOG" 2>&1
    echo "$(date -u) gated $tag block $b rc=$?" >> "$LOG"
  done
}

# calibration (1 block): bn128s1 vs bn128s1 must be 2.500
python3 e11_gate.py --cand "$BN" --ref "$BN" --seeds "$SEEDS" --workers "$WORKERS" \
    --seed0 500000 --out ckpt/aug/gates/calib_s500000.json >> "$LOG" 2>&1
echo "$(date -u) calib done" >> "$LOG"

# TTA on bn128s1 (needs only bn128s1) -> run first while nets may still be training
gate tta6 "$BN" --cand-tta 1 --tta-perms 0,1,2,3,4,5
gate tta3 "$BN" --cand-tta 1 --tta-perms 0,3,4

# wait for the 3 enhanced nets, then gate each
for seed in 0 1 2; do
  net="ckpt/aug/aug_128x40_s${seed}.pkl"
  while [ ! -f "$net" ] && [ ! -f /root/STOP_AUG ]; do sleep 120; done
  [ -f /root/STOP_AUG ] && break
  gate "aug_s${seed}" "$net"
done

python3 e11_agg.py >> "$LOG" 2>&1
echo "$(date -u) run_aug_gate DONE + aggregated" >> "$LOG"
