#!/bin/bash
# Waits (read-only) for the coordinator's 18 repl students, then runs the 6-block H1 replication.
# Does NOT launch/kill/modify the repl fleet or its pkls — only reads them.
cd /root/crossgame/doudizhu || exit 1
log(){ echo "[repl $(date +%F_%T)] $*"; }

log "waiting for 18 ckpt/students_repl/*.pkl AND repl fleet idle"
until [ "$(ls ckpt/students_repl/dou_rstudent_s*.pkl 2>/dev/null | wc -l)" -ge 18 ] && [ "$(pgrep -cf 'out ckpt/students_repl/dou_rstudent')" = "0" ]; do sleep 30; done
log "18 repl students present, fleet idle"

# pick a free GPU (memory-based; robust to PID namespace)
g=""
while [ -z "$g" ]; do
  for cand in 0 1 2 3 4 5 6 7; do
    used=$(nvidia-smi -i $cand --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ -z "$used" ] && continue
    if [ "$used" -lt 1500 ]; then g=$cand; break; fi
  done
  [ -z "$g" ] && sleep 20
done
log "running repl_h1 on GPU=$g"
CUDA_VISIBLE_DEVICES=$g python3 repl_h1.py --nblocks 6 --nseeds 2000 \
    --out results/noisy_h1_repl.json > logs/repl_h1.log 2>&1
log "REPL H1 DONE -> results/noisy_h1_repl.json"
