#!/bin/bash
# Full placement value-head run: 3 seeds on GPUs 0,1,2 (30k steps each).
# NEW files only. DONE markers + [ -f ] guards; loud failure.
set -u
cd /root/caiest_repro
mkdir -p ckpt/placeval results logs
for S in 0 1 2; do
  CK="ckpt/placeval/placeval_s${S}.pt"
  OUT="results/placeval_s${S}.json"
  LOG="logs/placeval_s${S}.log"
  if [ -f "$CK" ] && [ -f "$OUT" ]; then
    echo "seed $S already done ($CK) — skip"
    continue
  fi
  setsid bash -c "OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 \
      python3 placeval_train.py --seed $S --steps 30000 --gpu $S \
      --out $OUT --ckpt $CK > $LOG 2>&1; echo PLACEVAL_DONE_${S} >> $LOG" </dev/null >/dev/null 2>&1 &
  echo "launched seed $S on GPU $S -> $LOG"
done
echo "all launched"
