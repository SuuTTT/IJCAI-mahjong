#!/bin/bash
# 4070 keeper: v2 trainer (GPU) + v2 strength-eval loop (CPU). Cron every 5 min.
[ -f /root/T2_PAUSED ] || pgrep -f run_t2jax_v2_4070.sh >/dev/null 2>&1 || \
  { setsid nohup /root/run_t2jax_v2_4070.sh </dev/null >/dev/null 2>&1 & }
[ -f /root/T2_JAX_EVAL_PAUSED ] || pgrep -f '[r]un_t2jax_v2_eval.sh' >/dev/null 2>&1 || \
  { setsid nohup /root/run_t2jax_v2_eval.sh </dev/null >/dev/null 2>&1 & }
