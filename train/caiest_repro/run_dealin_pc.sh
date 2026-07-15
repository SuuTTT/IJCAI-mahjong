#!/bin/bash
# run_dealin_pc.sh -- 3 per-candidate deal-in seeds, one per GPU (1,2,3).
# GPUs 4,5,6 run the state-level model, GPU0 the pc smoke -> use 1,2,3. Idempotent (DONE guards).
cd /root/caiest_repro || exit 1
mkdir -p ckpt/dealin_pc logs
STEPS=50000
GPUS=(1 2 3)          # GPU for seed 0,1,2
for S in 0 1 2; do
  OUT="ckpt/dealin_pc/dealin_pc_s${S}.pt"
  DONE="ckpt/dealin_pc/dealin_pc_s${S}.DONE"
  LOG="logs/dealin_pc_s${S}.log"
  if [ -f "$DONE" ]; then echo "seed $S already DONE, skip"; continue; fi
  if [ -f "$OUT" ] && [ -f "${OUT}.traininfo.json" ]; then
    echo "seed $S ckpt+info exist, marking DONE"; touch "$DONE"; continue; fi
  G=${GPUS[$S]}
  echo "launching pc seed $S on GPU $G -> $OUT"
  ( CUDA_VISIBLE_DEVICES=$G python3 dealin_pc_train.py --seed "$S" --steps "$STEPS" \
        --val_every 2000 --out "$OUT" > "$LOG" 2>&1 && touch "$DONE" ) &
done
wait
echo "ALL PC SEEDS FINISHED $(date)"
