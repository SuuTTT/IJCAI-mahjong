#!/bin/bash
# Durable GPU watchdog — runs via cron every 10 min, session-independent.
# Slot 1: Boom league (always kept alive, re-arms its own 8h window).
# Slot 2: SMAX/Warpath job queue (/root/train_queue.txt), one at a time.
LOG=/root/gpu_keeper.log
echo "$(date +%H:%M) keeper tick" >> $LOG

# --- stall watchdog: catch a HUNG league (proc alive but GPU wedged, no gen
# progress) — the documented device_get hang. Conservative: 45-min staleness +
# idle GPU, well past a normal ~21-min gen + compile, so it never false-fires.
if [ ! -f /root/LEAGUE_PAUSED ] && pgrep -f run_league.sh > /dev/null; then
  LOGAGE=$(( $(date +%s) - $(stat -c %Y /root/league_run.log 2>/dev/null || echo 0) ))
  UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "$LOGAGE" -gt 2700 ] && [ "${UTIL:-99}" -lt 5 ]; then
    echo "$(date +%H:%M) league STALLED (${LOGAGE}s no progress, gpu ${UTIL}%) -> kill+rearm" >> $LOG
    pkill -9 -f ppo_selfplay; pkill -9 -f "baselines.league"; pkill -9 -f run_league.sh
    sleep 3
  fi
fi

# --- Slot 1: league ---
if [ ! -f /root/LEAGUE_PAUSED ] && ! pgrep -f run_league.sh > /dev/null; then
  sed -i "s/END=\$(( \$(date +%s) + [0-9]* ))/END=\$(( \$(date +%s) + 28800 ))/" /root/run_league.sh
  setsid nohup /root/run_league.sh > /root/league_win.log 2>&1 &
  echo "$(date +%H:%M) league re-armed" >> $LOG
fi

# --- Slot 1b: warpath (owns the GPU while the league is paused) ---
if [ -f /root/LEAGUE_PAUSED ] && ! pgrep -f run_warpath.sh > /dev/null; then
  setsid nohup /root/run_warpath.sh > /root/warpath_win.log 2>&1 &
  echo "$(date +%H:%M) warpath armed" >> $LOG
fi

# --- Slot 2: queue ---
Q=/root/train_queue.txt
if ! pgrep -f "QJOB_" > /dev/null; then
  # find first not-done line: format  name|command
  next=$(grep -vE "^#|^DONE" $Q 2>/dev/null | head -1)
  if [ -n "$next" ]; then
    name=$(echo "$next" | cut -d"|" -f1)
    cmd=$(echo "$next" | cut -d"|" -f2-)
    # mark started (prefix DONE so we never rerun it)
    sed -i "s|^$name|DONE $name|" $Q
    echo "$(date +%H:%M) launching $name" >> $LOG
    cd /root
    source /root/venv/bin/activate
    setsid bash -c "exec -a QJOB_$name env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.28 nice -n 10 python $cmd" > /root/qjob_$name.log 2>&1 &
  fi
fi

# --- Slot 3: Mahjong RL self-play (CPU, always on; coexists with the GPU league)
if [ ! -f /root/MAHJONG_RL_PAUSED ] && ! pgrep -f run_mahjong_rl.sh > /dev/null; then
  setsid nohup /root/run_mahjong_rl.sh > /root/mahjong_rl_win.log 2>&1 &
  echo "$(date +%H:%M) mahjong-rl armed" >> $LOG
fi

# --- Slot 4: Mahjong T2 RL fine-tune (GPU; only while Boom is paused)
if [ ! -f /root/T2_PAUSED ] && [ -f /root/LEAGUE_PAUSED ] && ! pgrep -f run_mahjong_t2.sh > /dev/null && ! pgrep -f mahjong_t2_jax > /dev/null; then
  setsid nohup /root/run_mahjong_t2.sh > /root/mahjong_t2_win.log 2>&1 &
  echo "$(date +%H:%M) mahjong-t2 armed" >> $LOG
fi

# --- Slot 4-JAX: fused on-GPU JAX KL-leashed PPO (supersedes torch-T2; GPU, only while Boom paused)
if [ ! -f /root/T2_JAX_PAUSED ] && [ -f /root/LEAGUE_PAUSED ] && ! pgrep -f 'run_mahjong_t2_jax.sh|baselines.mahjong_t2_jax' > /dev/null; then
  setsid nohup /root/run_mahjong_t2_jax.sh > /root/mahjong_t2_jax_win.log 2>&1 &
  echo "$(date +%H:%M) mahjong-t2-jax armed" >> $LOG
fi

# --- Slot 5: Mahjong T2-JAX STRENGTH-EVAL loop (CPU-only; always on; coexists with the GPU trainer)
# Snapshots latest.msgpack read-only, scores fine-tuned-vs-SL-anchor + pool, logs
# strength_eval.jsonl for the web monitor. NEVER touches the GPU or the trainer.
# NOTE: guarded by its OWN script name (run_mahjong_t2jax_eval.sh) which, unlike the
# trainer, does NOT contain the substring "mahjong_t2_jax" -> no pgrep cross-match.
if [ ! -f /root/T2_JAX_EVAL_PAUSED ] && ! pgrep -f run_mahjong_t2jax_eval.sh > /dev/null; then
  setsid nohup /root/run_mahjong_t2jax_eval.sh > /root/mahjong_t2jax_eval_win.log 2>&1 &
  echo "$(date +%H:%M) mahjong-t2jax-eval armed" >> $LOG
fi
