#!/bin/bash
cd /root/caiest_repro
exec python3 s2_oracle_gate.py --blocks 100 --seeds 80 --workers 100 --out results/SEARCH_ORACLE_FAST.json > logs/oracle_fast.log 2>&1
