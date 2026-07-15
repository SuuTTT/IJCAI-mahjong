#!/bin/bash
# Full grid: eps in {0.0,0.1,0.2,0.3,0.4} x seeds {0,1,2}.
# 3 GPUs (5,6,7), one sequential worker per GPU. New self-play dataset per seed
# (independent noisy observation of the coherent reference). Loud failure.
set -u
cd /root/poker_domain
GAMES=40000
EPS_LIST="0.0 0.1 0.2 0.3 0.4"
SEEDS="0 1 2"
N=6; M=3; EPOCHS=20
GPUS=(5 6 7)

echo "[driver] $(date -u) start" | tee -a driver.log

# ---- per-seed datasets (once, cheap, CPU) ----
for s in $SEEDS; do
  if [ ! -f data_s$s.npz ]; then
    echo "[driver] gen data seed $s" | tee -a driver.log
    python3 gen_data.py --ref reference_strategy.json --games $GAMES \
        --seed $s --out data_s$s.npz > gen_s$s.log 2>&1 || { echo "GEN FAIL s$s"; exit 1; }
  fi
done

# ---- build cell list and split across GPUs ----
rm -f gpu_5.list gpu_6.list gpu_7.list
i=0
for e in $EPS_LIST; do
  for s in $SEEDS; do
    g=${GPUS[$((i % 3))]}
    echo "$e $s" >> gpu_$g.list
    i=$((i+1))
  done
done

run_worker() {
  local g=$1
  local list=gpu_$g.list
  [ -f "$list" ] || return 0
  while read -r e s; do
    local cell="eps${e}_s${s}"
    local out="results/$cell"
    if [ -f "$out/CELL.DONE" ]; then
      echo "[gpu$g] skip $cell (done)" | tee -a driver.log
      continue
    fi
    echo "[gpu$g] $(date -u) start $cell" | tee -a driver.log
    CUDA_VISIBLE_DEVICES=$g python3 run_cell.py --data data_s$s.npz \
        --ref reference_strategy.json --eps $e --seed $s --outdir "$out" \
        --N $N --M $M --epochs $EPOCHS > cell_${cell}.log 2>&1 \
        || { echo "[gpu$g] CELL FAIL $cell" | tee -a driver.log; }
    echo "[gpu$g] $(date -u) end $cell" | tee -a driver.log
  done < "$list"
  touch WORKER_$g.DONE
}

mkdir -p results
for g in "${GPUS[@]}"; do
  run_worker $g &
done
wait
touch GRID.DONE
echo "[driver] $(date -u) GRID.DONE" | tee -a driver.log
python3 agg.py > agg.log 2>&1 && echo "[driver] aggregated" | tee -a driver.log
