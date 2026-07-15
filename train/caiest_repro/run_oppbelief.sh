#!/bin/bash
# run_oppbelief.sh -- 3 opponent-hand-belief seeds, one per GPU (0,1,2). Idempotent.
cd /root/caiest_repro || exit 1
mkdir -p ckpt/oppbelief logs
STEPS=50000
GPUS=(0 1 2)
for S in 0 1 2; do
  OUT="ckpt/oppbelief/oppbelief_s${S}.pt"
  DONE="ckpt/oppbelief/oppbelief_s${S}.DONE"
  LOG="logs/oppbelief_s${S}.log"
  if [ -f "$DONE" ]; then echo "seed $S DONE, skip"; continue; fi
  if [ -f "$OUT" ] && [ -f "${OUT}.traininfo.json" ]; then echo "seed $S exists, mark DONE"; touch "$DONE"; continue; fi
  G=${GPUS[$S]}
  echo "launch seed $S on GPU $G"
  ( CUDA_VISIBLE_DEVICES=$G python3 oppbelief_train.py --tag full --seed "$S" --steps "$STEPS" \
       --val_every 2000 --out "$OUT" > "$LOG" 2>&1 && touch "$DONE" ) &
done
wait
echo "OPPBELIEF ALL DONE $(date)"
