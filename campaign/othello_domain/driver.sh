#!/bin/bash
# Full Othello distill-then-ensemble grid.
# Cells: teacher depth D in {1,2,4,6} x eps in {0.0, 0.2}. GPUs 4-7 only.
# Data generated once per depth (eps applied at train time).
# Loud failure; DONE markers + [ -f ] guards; setsid-detached cells.
set -u
cd /root/othello_domain
BASE=/root/othello_domain
RESULTS=$BASE/results
mkdir -p $RESULTS
LOG=$BASE/driver.log
echo "[driver] start $(date)" | tee -a $LOG

NSTATES=100000
GENWORKERS=24
N=6; M=3; EPOCHS=18; NPAIRS=100; CH=64
EVALW=5

# ---- data generation (sequential per depth; load-friendly) ----
for D in 1 2 4 6; do
  DF=$BASE/data_d${D}.npz
  if [ -f "$DF" ]; then
    echo "[driver] data D=$D exists, skip" | tee -a $LOG
  else
    echo "[driver] gen data D=$D $(date)" | tee -a $LOG
    OMP_NUM_THREADS=1 python3 gen_data.py --depth $D --n_states $NSTATES \
       --workers $GENWORKERS --seed $((4000+D)) --out $DF >> $BASE/gen_d${D}.log 2>&1
    if [ ! -f "$DF" ]; then echo "[driver] FATAL gen D=$D failed" | tee -a $LOG; exit 1; fi
  fi
done
echo "[driver] data generation complete $(date)" | tee -a $LOG

# ---- cell runner: launches one cell detached on a GPU ----
run_cell () {
  local D=$1 EPS=$2 GPU=$3
  local tag="D${D}_eps${EPS}"
  local out=$RESULTS/$tag
  if [ -f "$out/CELL.json" ]; then echo "[driver] $tag done, skip" | tee -a $LOG; return; fi
  echo "[driver] launch $tag on GPU$GPU $(date)" | tee -a $LOG
  setsid bash -c "CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 python3 run_cell.py \
     --data $BASE/data_d${D}.npz --depth $D --eps $EPS --outdir $out \
     --N $N --M $M --epochs $EPOCHS --n_pairs $NPAIRS --ch $CH \
     --ladder 1 3 5 --eval_workers $EVALW --seed 0 \
     > $BASE/cell_${tag}.log 2>&1; touch $out/CELL.DONE" < /dev/null > /dev/null 2>&1 &
}

# ---- waves: one eps at a time; 4 depths across GPUs 4,5,6,7 ----
for EPS in 0.0 0.2; do
  echo "[driver] === WAVE eps=$EPS $(date) ===" | tee -a $LOG
  run_cell 1 $EPS 4
  run_cell 2 $EPS 5
  run_cell 4 $EPS 6
  run_cell 6 $EPS 7
  # wait for all four DONE markers
  for D in 1 2 4 6; do
    tag="D${D}_eps${EPS}"
    while [ ! -f "$RESULTS/$tag/CELL.DONE" ]; do sleep 15; done
    echo "[driver] $tag finished $(date)" | tee -a $LOG
  done
done

echo "[driver] all cells done, aggregating $(date)" | tee -a $LOG
python3 agg_othello.py --results $RESULTS --out $BASE/OTHELLO_DOMAIN.json >> $LOG 2>&1
echo "[driver] ALL DONE $(date)" | tee -a $LOG
touch $BASE/GRID.DONE
