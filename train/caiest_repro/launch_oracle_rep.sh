#!/bin/bash
cd /root/caiest_repro
# 3 more disjoint blocks (101,102,103 -> seed0 8.8M+b*3000) for a 4-block CI on the oracle ceiling
exec python3 s2_oracle_gate.py --blocks 101,102,103 --seeds 80 --workers 110 --out results/SEARCH_ORACLE_REP.json > logs/oracle_rep.log 2>&1
