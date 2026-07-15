#!/bin/bash
# Replication driver: run additional seeds for all 8 cells, reusing existing data.
# Usage: bash driver_multi.sh "1 2"   (space-separated seed list)
set -u
cd /root/othello_domain
BASE=/root/othello_domain
RESULTS=$BASE/results
LOG=$BASE/driver_multi.log
SEEDS="${1:-1 2}"
N=6; M=3; EPOCHS=18; NPAIRS=100; CH=64; EVALW=5
echo "[multi] start seeds=[$SEEDS] $(date)" | tee -a $LOG

run_cell () {
  local D=$1 EPS=$2 GPU=$3 SEED=$4
  local tag="D${D}_eps${EPS}_s${SEED}"
  local out=$RESULTS/$tag
  if [ -f "$out/CELL.json" ]; then echo "[multi] $tag done, skip" | tee -a $LOG; return; fi
  echo "[multi] launch $tag GPU$GPU $(date)" | tee -a $LOG
  setsid bash -c "CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 python3 run_cell.py \
     --data $BASE/data_d${D}.npz --depth $D --eps $EPS --outdir $out \
     --N $N --M $M --epochs $EPOCHS --n_pairs $NPAIRS --ch $CH \
     --ladder 1 3 5 --eval_workers $EVALW --seed $SEED \
     > $BASE/cell_${tag}.log 2>&1; touch $out/CELL.DONE" < /dev/null > /dev/null 2>&1 &
}

for SEED in $SEEDS; do
  for EPS in 0.0 0.2; do
    echo "[multi] === WAVE seed=$SEED eps=$EPS $(date) ===" | tee -a $LOG
    run_cell 1 $EPS 4 $SEED
    run_cell 2 $EPS 5 $SEED
    run_cell 4 $EPS 6 $SEED
    run_cell 6 $EPS 7 $SEED
    for D in 1 2 4 6; do
      tag="D${D}_eps${EPS}_s${SEED}"
      while [ ! -f "$RESULTS/$tag/CELL.DONE" ]; do sleep 15; done
      echo "[multi] $tag finished $(date)" | tee -a $LOG
    done
  done
done
echo "[multi] ALL SEEDS DONE $(date)" | tee -a $LOG
touch $BASE/MULTI.DONE
