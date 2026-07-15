#!/bin/bash
# 7-config LEAGUE / opponent-pool self-play sweep on GPUs 1-7 (GPU0 left for g0_control
# anchor-baseline reference). Each config: setsid-detached, timeout 7h, B=256 bf16,
# PopArt on, kdens3 warm-start, frozen-kdens3-anchor strength-eval JSONL (shared yardstick).
# Trainer: baselines/mahjong_t2_jax_v3_league.py  (v2 + opponent pool + KL anneal).
set -u
cd /root/ludus_rl
PY=/usr/bin/python3
ROOT=/root/rl_sweep
TL=25200   # 7h

COMMON="--updates 100000 --B 256 --T 256 --K 20480 --minibatch 1024 --epochs 3 \
--dtype bf16 --popart 1 --value-epochs 6 --lr-pi 3e-5 --lr-v 3e-4 \
--pool-refresh 25 --p-anchor 0.3 \
--eval-games 2000 --eval-every-updates 25 --eval-at-start --ckpt-every-sec 1200 --log-every 1"

# name  gpu  extra-flags
run_cfg () {
  local name="$1" gpu="$2"; shift 2
  local out="$ROOT/league_${name}/"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" setsid bash -c \
    "cd /root/ludus_rl && timeout $TL $PY -m baselines.mahjong_t2_jax_v3_league $COMMON $* --out '$out' --resume > '$out/run.log' 2>&1" &
  echo "launched $name on GPU$gpu -> $out"
}

# G1: pool-cap 10, KL-anneal 0.05->0.30
run_cfg g1_pc10_kl0530        1  --seed 1 --pool-cap 10 --kl-target 0.05 --kl-target-final 0.30 --kl-anneal-updates 2000 --entcoef 0.001
# G2: pool-cap 20, KL-anneal 0.05->0.30
run_cfg g2_pc20_kl0530        2  --seed 1 --pool-cap 20 --kl-target 0.05 --kl-target-final 0.30 --kl-anneal-updates 2000 --entcoef 0.001
# G3: pool-cap 10, KL fixed 0.10 (isolate the pool effect)
run_cfg g3_pc10_klfix10       3  --seed 1 --pool-cap 10 --kl-target 0.10 --kl-target-final 0.10 --kl-anneal-updates 0    --entcoef 0.001
# G4: pool-cap 10, KL-anneal 0.05->0.50, entropy 0.02 (league + escape-basin)
run_cfg g4_pc10_kl0550_ent02  4  --seed 1 --pool-cap 10 --kl-target 0.05 --kl-target-final 0.50 --kl-anneal-updates 2000 --entcoef 0.02
# G5: pool-cap 5, KL-anneal 0.05->0.30
run_cfg g5_pc5_kl0530         5  --seed 1 --pool-cap 5  --kl-target 0.05 --kl-target-final 0.30 --kl-anneal-updates 2000 --entcoef 0.001
# G6: pool-cap 20, KL-anneal 0.05->0.50, entropy 0.02
run_cfg g6_pc20_kl0550_ent02  6  --seed 1 --pool-cap 20 --kl-target 0.05 --kl-target-final 0.50 --kl-anneal-updates 2000 --entcoef 0.02
# G7: pool-cap 10, KL-anneal 0.05->0.30, seed 2
run_cfg g7_pc10_kl0530_s2     7  --seed 2 --pool-cap 10 --kl-target 0.05 --kl-target-final 0.30 --kl-anneal-updates 2000 --entcoef 0.001

sleep 2
echo "=== all league configs launched ==="
