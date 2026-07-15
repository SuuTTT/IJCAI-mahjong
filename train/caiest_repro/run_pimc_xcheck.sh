#!/bin/bash
cd /root/caiest_repro
exec python3 s5_pimc_gate.py --blocks 0 --seeds 16 --true_state --n_worlds 1 --workers 28 --out /tmp/pimc_oracle_xcheck.json > /tmp/pimc_xcheck.log 2>&1
