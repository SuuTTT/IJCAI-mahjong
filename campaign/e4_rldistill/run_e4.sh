#!/bin/bash
# E4 RL policy distillation probe — full chain driver.
# Loud failure: set -e stops at the first crashing stage; the error stays in
# that stage's log. DONE markers + [ -f ] guards make re-runs resume.
set -euo pipefail
cd /root/e4_rldistill
export CUDA_VISIBLE_DEVICES=7
PY=/root/e4_rldistill/venv/bin/python

# Overridable for smoke tests
CKPT=${CKPT_DIR:-ckpt}
RESULTS=${RESULTS_DIR:-results}
LOGS=${LOGS_DIR:-logs}
STEPS=${STEPS:-3000000}
BUFFER_STATES=${BUFFER_STATES:-100000}
STUDENT_EPOCHS=${STUDENT_EPOCHS:-20}
NEP_TEACHER=${NEP_TEACHER:-100}
NEP_FINAL=${NEP_FINAL:-200}
mkdir -p "$CKPT" "$RESULTS" "$LOGS"

echo "[driver $(date -u +%FT%TZ)] E4 chain start (STEPS=$STEPS BUFFER=$BUFFER_STATES)"

# ---- Stage 1: teachers (PPO, seeds 0-5, sequential on GPU 7) ----
for s in 0 1 2 3 4 5; do
  if [ -f "$CKPT/DONE_teacher_s$s" ]; then
    echo "[driver] teacher s$s already done, skipping"
  else
    echo "[driver $(date -u +%FT%TZ)] training teacher s$s ..."
    $PY src/ppo_train.py --seed $s --total-steps $STEPS \
        --out "$CKPT/teacher_s$s.pt" > "$LOGS/teacher_s$s.log" 2>&1
    touch "$CKPT/DONE_teacher_s$s"
  fi
done

# ---- Stage 2: per-teacher solo eval (100 eps, greedy, seeds 10000+) ----
for s in 0 1 2 3 4 5; do
  if [ -f "$CKPT/DONE_evalteacher_s$s" ]; then
    echo "[driver] teacher eval s$s already done, skipping"
  else
    echo "[driver $(date -u +%FT%TZ)] evaluating teacher s$s ..."
    $PY src/eval_teacher.py --seed $s --ckpt "$CKPT/teacher_s$s.pt" \
        --out "$RESULTS/teacher_s$s.json" --n-episodes $NEP_TEACHER \
        --eval-seed-start 10000 > "$LOGS/eval_teacher_s$s.log" 2>&1
    touch "$CKPT/DONE_evalteacher_s$s"
  fi
done

# ---- Stage 3: mixture state buffer (~100k states, all 6 teachers) ----
if [ -f "$CKPT/DONE_buffer" ]; then
  echo "[driver] buffer already done, skipping"
else
  echo "[driver $(date -u +%FT%TZ)] building state buffer ..."
  $PY src/build_buffer.py --ckpt-dir "$CKPT" --n-states $BUFFER_STATES \
      --out "$CKPT/buffer.npz" > "$LOGS/buffer.log" 2>&1
  touch "$CKPT/DONE_buffer"
fi

# ---- Stage 4: students (KL to mean teacher distribution, seeds 10-12) ----
for s in 10 11 12; do
  if [ -f "$CKPT/DONE_student_s$s" ]; then
    echo "[driver] student s$s already done, skipping"
  else
    echo "[driver $(date -u +%FT%TZ)] training student s$s ..."
    $PY src/train_student.py --seed $s --buffer "$CKPT/buffer.npz" \
        --epochs $STUDENT_EPOCHS --out "$CKPT/student_s$s.pt" \
        > "$LOGS/student_s$s.log" 2>&1
    touch "$CKPT/DONE_student_s$s"
  fi
done

# ---- Stage 5: final eval (the verdict) ----
if [ -f "$CKPT/DONE_final" ]; then
  echo "[driver] final eval already done, skipping"
else
  echo "[driver $(date -u +%FT%TZ)] final eval ..."
  $PY src/final_eval.py --ckpt-dir "$CKPT" --results-dir "$RESULTS" \
      --n-episodes $NEP_FINAL --eval-seed-start 20000 \
      --out "$RESULTS/E4_RLDISTILL.json" > "$LOGS/final_eval.log" 2>&1
  touch "$CKPT/DONE_final"
fi

echo "[driver $(date -u +%FT%TZ)] E4 ALL DONE — verdict in $RESULTS/E4_RLDISTILL.json"
