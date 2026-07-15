#!/bin/bash
# Wait until the synth-hard grid finishes AND no domain jobs remain, THEN run the
# oracle replication ALONE (lesson: oracle 110-worker CPU job cannot coexist w/ GPU trainings).
cd /root/caiest_repro
for i in $(seq 1 240); do
  synthdone=0; [ -f /root/synth_coherence/SYNTH_HARD_GRID_DONE ] && synthdone=1
  nproc_dom=$(ps -C python3 -o cmd --no-headers 2>/dev/null | grep -cE "synth_hard|run_cell|gen_data")
  if [ "$synthdone" = "1" ] && [ "$nproc_dom" -eq 0 ]; then break; fi
  sleep 30
done
# now run oracle rep alone: 2 disjoint blocks, 70 workers (box is free)
python3 s2_oracle_gate.py --blocks 101,102 --seeds 60 --workers 70 --out results/SEARCH_ORACLE_REP.json > logs/oracle_rep_alone.log 2>&1
touch results/ORACLE_REP_DONE
