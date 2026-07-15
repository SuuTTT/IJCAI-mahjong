#!/bin/bash
# Durable launcher for the T2-JAX v2 Mahjong trainer on the RTX 4070 (12GB -> B=256 bf16).
# v2 vs baseline: PopArt-lite return normalization (--popart 1) + extra value-only epochs
# (--value-epochs 6) so the fresh critic converges (raw vloss ~0.02 vs baseline flat ~1.6-2.0).
# Loops with --resume so a crash/OOM re-attaches to the latest.msgpack checkpoint.
# This mirrors the box copy at /root/run_t2jax_v2_4070.sh (the keeper relaunches this wrapper).
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd /root/ludus
source /root/venv/bin/activate
export PYTHONPATH=/root/ludus
OUT=/root/ludus_train/mahjong_t2_jax_v2
mkdir -p "$OUT"
while true; do
  python -m baselines.mahjong_t2_jax_v2 \
      --seed 1 --updates 100000 --out "$OUT/" \
      --B 256 --T 256 --K 20480 --dtype bf16 \
      --kl-target 0.05 --beta0 0.08 \
      --popart 1 --value-epochs 6 \
      --eval-games 2000 --eval-every-updates 25 --eval-at-start \
      --ckpt-every-sec 600 --log-every 1 --resume >> /root/mahjong_t2_jax_v2.log 2>&1
  echo "[wrapper] exited $? at $(date), restart in 20s" >> /root/mahjong_t2_jax_v2.log
  sleep 20
done
