#!/bin/bash
# Robustness probe: does stronger KD (alpha) change the distill-vs-ensemble story?
# Run pure-distillation (alpha=1.0) and alpha=0.9 at two noise levels, 3 seeds.
set -u
cd /root/poker_domain
GPUS=(5 6 7)
N=6; M=3; EPOCHS=20
i=0
for al in 0.9 1.0; do
  for e in 0.2 0.4; do
    for s in 0 1 2; do
      g=${GPUS[$((i % 3))]}
      cell="a${al}_eps${e}_s${s}"
      out="results_alpha/$cell"
      [ -f "$out/CELL.DONE" ] && { i=$((i+1)); continue; }
      CUDA_VISIBLE_DEVICES=$g python3 run_cell.py --data data_s$s.npz \
          --ref reference_strategy.json --eps $e --seed $s --outdir "$out" \
          --alpha $al --N $N --M $M --epochs $EPOCHS > cell_${cell}.log 2>&1 &
      i=$((i+1))
      # throttle: at most 3 concurrent (one per gpu)
      if [ $((i % 3)) -eq 0 ]; then wait; fi
    done
  done
done
wait
touch ALPHA.DONE
echo "alpha probe done"
