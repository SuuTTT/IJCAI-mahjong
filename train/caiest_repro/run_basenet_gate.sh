#!/bin/bash
# BASENET gate harness: gate each candidate BASE net (raw policy, lam=0) vs the deployed
# distill net (cnn_lad_chunjiandu.npz) in the calibrated duplicate-format placement gate.
# Calibration: distill-vs-distill must read 2.5000. A net "beats distill" only if its
# gate-vs-distill 95% CI lower bound is strictly > 2.500.
# Free-guarded (honors /root/STOP_BNG), idempotent (skips existing cells), CPU-only, good neighbor.
cd /root/IJCAI-mahjong/train/caiest_repro || exit 1
REF=/root/assets/cnn_lad_chunjiandu.npz
RKIND=resbn_fused; RCFG="channels=128,blocks=40"
OUT=ckpt/basenet_gate; mkdir -p "$OUT"
LOG=/root/basenet_gate.log
SEEDS=300; W=56
SEED0S="70000 80000 90000 100000 110000"
echo "$(date -u) BASENET_GATE START" >> "$LOG"

# label|path|kind|cfg   (calib_distill first as the 2.500 sanity check)
CANDS=(
 "calib_distill|/root/assets/cnn_lad_chunjiandu.npz|resbn_fused|channels=128,blocks=40"
 "moyu_bn_128x40|/root/assets/moyu_bn_128x40.pkl|resbn|channels=128,blocks=40"
 "full_128x40_s0|ckpt/e1b/full_128x40_s0.pkl|resbn_fused|channels=128,blocks=40"
 "full_128x40_s1|ckpt/e1b/full_128x40_s1.pkl|resbn_fused|channels=128,blocks=40"
 "full_256x40_s0|ckpt/e1b/full_256x40_s0.pkl|resbn_fused|channels=256,blocks=40"
 "full_256x40_s1|ckpt/e1b/full_256x40_s1.pkl|resbn_fused|channels=256,blocks=40"
 "full_384x40_s0|ckpt/e1b/full_384x40_s0.pkl|resbn_fused|channels=384,blocks=40"
 "big192x40_s0_fused|ckpt/big192x40_s0_fused.pkl|resbn_fused|channels=192,blocks=40"
 "big256x40_s0_fused|ckpt/big256x40_s0_fused.pkl|resbn_fused|channels=256,blocks=40"
)

for entry in "${CANDS[@]}"; do
  IFS='|' read -r lbl path kind cfg <<< "$entry"
  if [ ! -f "$path" ]; then echo "$(date -u) SKIP missing $lbl ($path)" >> "$LOG"; continue; fi
  for s0 in $SEED0S; do
    [ -f /root/STOP_BNG ] && { echo "$(date -u) STOP_BNG -> halt" >> "$LOG"; exit 0; }
    gj="$OUT/${lbl}_s${s0}.json"
    [ -f "$gj" ] && continue
    python3 e8_gate.py --cand "$path" --cand-kind "$kind" --cand-cfg "$cfg" \
      --ref "$REF" --ref-kind "$RKIND" --ref-cfg "$RCFG" \
      --lam 0 --seeds "$SEEDS" --workers "$W" --seed0 "$s0" --out "$gj" \
      >> "$LOG" 2>&1 && echo "$(date -u) GATE done $lbl s$s0" >> "$LOG" \
      || echo "$(date -u) GATE FAIL $lbl s$s0" >> "$LOG"
  done
done
echo "$(date -u) BASENET_GATE ALL DONE" >> "$LOG"
