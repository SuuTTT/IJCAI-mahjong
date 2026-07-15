#!/bin/bash
cd /root/caiest_repro
# lightweight cross-block confirmation of the 3.55 ceiling: 2 disjoint blocks, 40 workers
# (kept small to coexist with GPU domain jobs without CPU oversubscription)
exec python3 s2_oracle_gate.py --blocks 101,102 --seeds 40 --workers 40 --out results/SEARCH_ORACLE_REP.json > logs/oracle_rep2.log 2>&1
