#!/bin/bash
# run_dv_grid.sh -- value-aware action-value defense: lambda x K grid vs kdens3.
# 4 blocks x 500 seeds each. Idempotent per cell. GPU-assisted, 48 workers.
cd /root/caiest_repro || exit 1
mkdir -p results logs
W=30
run(){  # $1=lam $2=K $3=name
  OUT="results/DV_cell_$3.json"
  if [ -f "$OUT" ]; then echo "cell $3 exists, skip"; return; fi
  echo "=== cell $3 (lam=$1 K=$2) start $(date)"
  python3 s4_defense_value_gate.py --blocks 0,1,2,3 --seeds 500 --workers $W --lam "$1" --K "$2" \
     --out "$OUT" > "logs/dv_$3.log" 2>&1
  echo "=== cell $3 done $(date)"
}
run 0   3   lam0_K3
run 0.5 3   lam0.5_K3
run 1   3   lam1_K3
run 2   3   lam2_K3
run 4   3   lam4_K3
run 0   -1  lam0_Kall
echo "DV GRID DONE $(date)"
