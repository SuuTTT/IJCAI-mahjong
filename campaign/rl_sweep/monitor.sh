#!/bin/bash
# refresh the status MD every 15 min for ~9h then exit
for i in $(seq 1 36); do
  python3 /root/rl_sweep/summarize.py > /root/rl_sweep/results/last_summary.txt 2>&1
  sleep 900
done
