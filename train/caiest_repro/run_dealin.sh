#!/bin/bash
# run_dealin.sh -- launch 3 deal-in predictor seeds, one per GPU (4,5,6).
# Per-seed DONE markers + [ -f ] guards make it idempotent/re-runnable.
cd /root/caiest_repro || exit 1
mkdir -p ckpt/dealin logs
STEPS=50000
GPUS=(4 5 6)          # GPU for seed 0,1,2
for S in 0 1 2; do
  OUT="ckpt/dealin/dealin_s${S}.pt"
  DONE="ckpt/dealin/dealin_s${S}.DONE"
  LOG="logs/dealin_s${S}.log"
  if [ -f "$DONE" ]; then echo "seed $S already DONE, skip"; continue; fi
  if [ -f "$OUT" ] && [ -f "${OUT}.traininfo.json" ]; then
    echo "seed $S ckpt+info exist, marking DONE"; touch "$DONE"; continue; fi
  G=${GPUS[$S]}
  echo "launching seed $S on GPU $G -> $OUT"
  ( CUDA_VISIBLE_DEVICES=$G python3 dealin_train.py --seed "$S" --steps "$STEPS" \
        --val_every 2000 --out "$OUT" > "$LOG" 2>&1 && touch "$DONE" ) &
done
wait
echo "ALL SEEDS FINISHED $(date)"
