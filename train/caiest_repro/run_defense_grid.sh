#!/bin/bash
# run_defense_grid.sh -- defensive-policy tau/K grid vs kdens3. 4 blocks x 500 seeds each.
# Idempotent per cell ([ -f ] guard). Sequential to keep <=128 cores.
cd /root/caiest_repro || exit 1
mkdir -p results logs
W=120
run(){  # $1=tau $2=K $3=name
  OUT="results/DEFENSE_cell_$3.json"
  if [ -f "$OUT" ]; then echo "cell $3 exists, skip"; return; fi
  echo "=== cell $3 (tau=$1 K=$2) start $(date)"
  python3 s3_defense_gate.py --blocks 0,1,2,3 --seeds 500 --workers $W --tau "$1" --K "$2" --out "$OUT" \
    > "logs/def_$3.log" 2>&1
  echo "=== cell $3 done $(date)"
}
run 0.3 3   tau0.3_K3
run 0.5 3   tau0.5_K3
run 0.7 3   tau0.7_K3
run 0.0 3   tau0.0_K3
run 0.0 -1  tau0.0_Kall
echo "GRID DONE $(date)"
