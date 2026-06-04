#!/bin/bash
# Auto-restarting league trainer for unattended 8h run. Checkpoints to --out every 25 iters,
# so a restart resumes from the latest pool snapshot anchor (base) with little loss.
cd "$(dirname "$0")"
LOG=/tmp/league_run.log
OUT=arch_ck/explore/resbn40_league.pkl
END=$(( $(date +%s) + 8*3600 ))   # stop launching after 8h
echo "=== league launcher start $(date) ===" >> $LOG
while [ "$(date +%s)" -lt "$END" ]; do
  # resume from the best-so-far checkpoint if it exists, else the SL base
  BASE=arch_ck/explore/resbn40.pkl
  if [ -f "$OUT" ]; then BASE="$OUT"; echo "[$(date +%T)] resuming from $OUT" >> $LOG; fi
  OPENBLAS_NUM_THREADS=1 python3 rl_league.py --base "$BASE" --blocks 40 --iters 400 \
      --main-actors 16 --exp-actors 6 --games-per-actor 3 \
      --snap-every 10 --eval-every 25 --out "$OUT" >> $LOG 2>&1
  echo "[$(date +%T)] python exited code $? — restarting in 10s" >> $LOG
  sleep 10
done
echo "=== league launcher done (8h elapsed) $(date) ===" >> $LOG
