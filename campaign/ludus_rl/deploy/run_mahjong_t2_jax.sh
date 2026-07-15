#!/bin/bash
# Durable launcher for the fused on-GPU JAX KL-leashed PPO Mahjong trainer (T2-JAX).
# Loops with --resume so a crash/OOM re-attaches to the latest.msgpack checkpoint.
# REQUIRED cuDNN fix for JAX-GPU is exported here.
set -u
export LD_LIBRARY_PATH=/tmp/cudnn98/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
cd /root/ludus
OUT=/root/ludus_train/mahjong_t2_jax
mkdir -p "$OUT"
PY=/root/venv/bin/python
while true; do
  $PY -m baselines.mahjong_t2_jax \
      --seed 1 --updates 100000 \
      --out "$OUT/" \
      --B 512 --T 256 --K 20480 --dtype bf16 \
      --kl-target 0.05 --beta0 0.08 \
      --eval-games 2000 --eval-every-updates 25 --eval-at-start \
      --ckpt-every-sec 600 --log-every 1 \
      --resume >> /root/mahjong_t2_jax.log 2>&1
  echo "[wrapper] trainer exited code $? at $(date), restarting in 20s" >> /root/mahjong_t2_jax.log
  sleep 20
done
