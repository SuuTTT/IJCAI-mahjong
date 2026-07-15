#!/bin/bash
cd /root/caiest_repro
# FIRST real PIMC gate: uniform determinization, N=20 worlds, value-cutoff K=6, 2 blocks x 8 seeds.
# modest workers (box is CPU-contended by belief/dealin gens+trains).
exec python3 s6_pimc_vcut.py --blocks 0,1 --seeds 8 --n_worlds 20 --k_cutoff 6 --workers 22 --out results/PIMC_GATE1_uniform.json > logs/pimc_gate1.log 2>&1
