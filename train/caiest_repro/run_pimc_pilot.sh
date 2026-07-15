#!/bin/bash
cd /root/caiest_repro
# first-signal PIMC pilot: uniform determinization, N=8 worlds, 2 blocks x 4 seeds.
# light (16 workers) so it does not starve the belief self-play gen.
exec python3 s5_pimc_gate.py --blocks 0,1 --seeds 4 --n_worlds 8 --workers 16 --out results/PIMC_PILOT_N8.json > logs/pimc_pilot_n8.log 2>&1
