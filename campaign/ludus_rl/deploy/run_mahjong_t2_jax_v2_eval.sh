#!/bin/bash
# Durable STRENGTH-EVAL loop for the T2-JAX Mahjong fine-tune (CPU-only).
#
# Every ~EVERY_SEC it snapshots the CURRENT latest.msgpack (read-only -- never
# touches the trainer), evaluates it (fine-tuned vs SL-anchor headline + vs
# EfficiencyBot/random + a small kdens3 sample), appends one JSONL line the web
# monitor charts, and keeps a running best. It NEVER uses the GPU and NEVER
# disturbs the trainer.
#
# Guards:
#   * pause file /root/T2_JAX_EVAL_PAUSED disables the loop.
#   * single-instance: the [r]un_... pgrep trick (self-exclusion) so a 2nd copy
#     launched by the keeper exits immediately.
# Local source of truth: ludus repo deploy/run_mahjong_t2_jax_eval.sh
set -u
export JAX_PLATFORMS=cpu                         # CPU ONLY -- the GPU is the trainer's
export MCR_CHAMPION_DIR=/root/mcr_champion
export MCR_SL_ANCHOR=/root/mcr_champion/kdens_s0_fp16.npz
export PYTHONPATH=/root/ludus:/root/ludus/baselines

OUT=/root/ludus_train/mahjong_t2_jax_v2
LATEST=$OUT/latest.msgpack
SNAP=$OUT/_eval_snapshot.msgpack
JSONL=$OUT/strength_eval.jsonl
BEST=$OUT/best_by_eval.msgpack
RESULTS=$OUT/seed1_jax_results.jsonl
LOG=/root/mahjong_t2_jax_v2_eval.log
PY=/root/venv/bin/python
EVERY_SEC=${EVERY_SEC:-1500}                     # ~25 min between evals

# eval budget (games per matchup) -- tune here; all counts are logged, never truncated
N_ANCHOR=${N_ANCHOR:-40}
N_ANCHOR_SELF=${N_ANCHOR_SELF:-30}
N_EFF=${N_EFF:-30}
N_RAND=${N_RAND:-30}
N_KDENS3=${N_KDENS3:-6}
SEED=${SEED:-10000}

# --- single-instance guard (self-excluding [r]un_ pattern) ---
if [ "$(pgrep -fc '[r]un_t2jax_v2_eval.sh')" -gt 1 ]; then
  echo "$(date '+%F %T') another eval loop is running -> exit" >> "$LOG"
  exit 0
fi

echo "$(date '+%F %T') eval loop START (every ${EVERY_SEC}s, CPU-only)" >> "$LOG"
cd /root/ludus

while true; do
  if [ -f /root/T2_JAX_EVAL_PAUSED ]; then
    echo "$(date '+%F %T') paused (T2_JAX_EVAL_PAUSED)" >> "$LOG"
    sleep 120; continue
  fi
  if [ ! -f "$LATEST" ]; then
    echo "$(date '+%F %T') no checkpoint yet" >> "$LOG"
    sleep 120; continue
  fi

  # snapshot the checkpoint (trainer may overwrite latest.msgpack mid-eval)
  cp -f "$LATEST" "$SNAP" 2>>"$LOG"
  # validate the snapshot is a loadable msgpack (skip a torn read)
  if ! $PY -c "import flax.serialization as f; f.msgpack_restore(open('$SNAP','rb').read())" >/dev/null 2>>"$LOG"; then
    echo "$(date '+%F %T') snapshot unreadable (torn write?) -> retry in 60s" >> "$LOG"
    sleep 60; continue
  fi

  TS=$(date +%s)                                  # ts_unix from the SHELL, not python
  # env_steps from the trainer's latest jsonl line (shell reads it, passes it in)
  ENVSTEPS=$($PY -c "import json;L=[l for l in open('$RESULTS') if l.strip()];print(json.loads(L[-1])['env_steps'] if L else 0)" 2>>"$LOG" || echo 0)

  echo "$(date '+%F %T') eval snapshot ts=$TS env_steps=$ENVSTEPS" >> "$LOG"
  $PY -m baselines.mahjong_t2jax_strength \
      --ckpt "$SNAP" --anchor "$MCR_SL_ANCHOR" \
      --ts "$TS" --env-steps "$ENVSTEPS" \
      --n-anchor $N_ANCHOR --n-anchor-self $N_ANCHOR_SELF \
      --n-eff $N_EFF --n-rand $N_RAND --n-kdens3 $N_KDENS3 --seed $SEED \
      --out-jsonl "$JSONL" --best-ckpt "$BEST" >> "$LOG" 2>&1
  echo "$(date '+%F %T') eval done (exit $?)" >> "$LOG"

  sleep "$EVERY_SEC"
done
