#!/bin/bash
# Ready-to-launch full LEAGUE / opponent-pool self-play run (T2-JAX v3).
# Complementary lever to the F3 wall: opponents come from a growing pool of learner
# snapshots + the frozen kdens3 anchor (not frozen-self), with a tight->loose KL anneal.
# The strength EVAL still plays seat-0 vs the FROZEN kdens3 anchor -- same yardstick as v2
# and the wide-explore sweep, so results are directly comparable.
#
# Schedule this on a FREED, dedicated GPU (needs ~11 GB at B=256 bf16; the shared box's
# per-GPU 3.6 GB free is NOT enough -- wait for the sweep to release a card). On a
# dedicated GPU leave XLA autotune ON (default) for full throughput.
#
# Usage:  CUDA_VISIBLE_DEVICES=<gpu> bash deploy/run_mahjong_t2_jax_v3_league.sh
set -euo pipefail
cd /root/ludus_rl
PY=/usr/bin/python3
OUT=/root/ludus_train/mahjong_t2_jax_v3_league_s1/
mkdir -p "$OUT"

exec $PY -m baselines.mahjong_t2_jax_v3_league \
  --seed 1 --updates 100000 --out "$OUT" --resume \
  --B 256 --T 256 --K 20480 --minibatch 1024 --epochs 3 \
  --dtype bf16 --popart 1 --value-epochs 6 \
  --lr-pi 3e-5 --lr-v 3e-4 --clip 0.2 --lam 0.95 --vcoef 0.5 --entcoef 0.001 \
  --beta0 0.08 --kl-gate-frac 0.7 \
  `# --- LEAGUE opponent pool ---` \
  --pool-cap 16 --pool-refresh 50 --p-anchor 0.3 \
  `# --- KL tight->loose anneal (F3 lever #1): 0.05 -> 0.20 over 3000 updates ---` \
  --kl-target 0.05 --kl-target-final 0.20 --kl-anneal-updates 3000 \
  `# --- frozen-anchor strength eval (the yardstick) ---` \
  --eval-games 2000 --eval-every-updates 25 --eval-at-start \
  --ckpt-every-sec 1200 --log-every 1
