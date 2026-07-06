#!/bin/bash
# E1b eval harness — for each FULLY-CONVERGED model in ckpt/e1b/<label>.pkl: claim-rate + expert-gap
# (e1_measure) and placement (e1_gate) raw (tau=0) and suppressed (tau=2), each over 3 seed-blocks
# (mean+/-std). Idempotent. Free-guarded measure; gates CPU-bound run in series. Waits for training.
cd /root/IJCAI-mahjong/train/caiest_repro || exit 1
LOG=/root/e1b_eval.log; MD=ckpt/e1b/meas; GD=ckpt/e1b/gates; mkdir -p "$MD" "$GD"
echo "$(date -u) e1b_eval START" >> "$LOG"
MOYU=/root/assets/moyu_bn_128x40.pkl
SEEDS=300; W=56
SEED0S="70000 80000 90000"
END=$(( $(date +%s) + 86400 ))

chan_of(){ case "$1" in *_64x*) echo 64;; *_256x*) echo 256;; *_384x*) echo 384;; *) echo 128;; esac; }
blk_of(){ case "$1" in *x6_*) echo 6;; *x20_*) echo 20;; *) echo 40;; esac; }
train_running(){ pgrep -f "e1b_train_harness.sh" >/dev/null || pgrep -f "e1_train.py --channels" >/dev/null; }

while [ "$(date +%s)" -lt "$END" ] && [ ! -f /root/STOP_E1B ]; do
  done_all=1; n_models=0
  for pkl in ckpt/e1b/*.pkl; do
    [ -e "$pkl" ] || continue
    case "$pkl" in *_probe*|*.bn.pkl) continue;; esac
    n_models=$(( n_models + 1 ))
    lbl=$(basename "$pkl" .pkl)
    ch=$(chan_of "$lbl"); blk=$(blk_of "$lbl"); cfg="channels=$ch,blocks=$blk"
    if [ ! -f "$MD/$lbl.json" ]; then
      done_all=0
      CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 python3 e1_measure.py --model "$pkl" --kind resbn_fused \
        --channels "$ch" --blocks "$blk" --gpu 0 --out "$MD/$lbl.json" >> "$LOG" 2>&1 \
        && echo "$(date -u) MEAS done $lbl" >> "$LOG" || echo "$(date -u) MEAS FAIL $lbl" >> "$LOG"
    fi
    for tau in 0 2; do
      for s0 in $SEED0S; do
        gj="$GD/${lbl}_tau${tau}_s${s0}.json"
        [ -f "$gj" ] && continue
        done_all=0
        python3 e1_gate.py --cand "$pkl" --cand-kind resbn_fused --cand-cfg "$cfg" \
          --ref "$MOYU" --ref-kind resbn --ref-cfg "channels=128,blocks=40" \
          --claim-tau "$tau" --seeds "$SEEDS" --workers "$W" --seed0 "$s0" --out "$gj" \
          >> "$LOG" 2>&1 && echo "$(date -u) GATE done $lbl tau$tau s$s0" >> "$LOG" \
          || echo "$(date -u) GATE FAIL $lbl tau$tau s$s0" >> "$LOG"
      done
    done
  done
  if [ "$done_all" -eq 1 ] && [ "$n_models" -ge 1 ] && ! train_running; then
    echo "$(date -u) e1b_eval ALL DONE (n_models=$n_models)" >> "$LOG"; break
  fi
  sleep 120
done
echo "$(date -u) e1b_eval END" >> "$LOG"
