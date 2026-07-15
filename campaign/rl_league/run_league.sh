#!/bin/bash
# LEAGUE self-play sweep — the F3 fix (opponent pool + KL-anneal). 8 configs, GPUs 0-7, ~7h each.
mkdir -p /root/rl_league
TR=/root/ludus_rl/baselines/mahjong_t2_jax_v3_league.py
COMMON="--B 256 --T 256 --dtype bf16 --updates 100000 --pool-refresh 25"
launch(){ local g=$1 name=$2; shift 2
  setsid bash -c "CUDA_VISIBLE_DEVICES=$g timeout 25200 python3 $TR $COMMON $* --out /root/rl_league/$name/ > /root/rl_league/$name.log 2>&1" < /dev/null > /dev/null 2>&1 & }
launch 0 L0_cap10_anneal   --pool-cap 10 --p-anchor 0.3 --kl-target 0.05 --kl-target-final 0.30 --kl-anneal-updates 400 --entcoef 0.001 --seed 1
launch 1 L1_cap20_anneal   --pool-cap 20 --p-anchor 0.3 --kl-target 0.05 --kl-target-final 0.30 --kl-anneal-updates 400 --entcoef 0.001 --seed 1
launch 2 L2_cap10_klfix10  --pool-cap 10 --p-anchor 0.3 --kl-target 0.10 --entcoef 0.001 --seed 1
launch 3 L3_cap10_esc      --pool-cap 10 --p-anchor 0.3 --kl-target 0.05 --kl-target-final 0.50 --kl-anneal-updates 400 --entcoef 0.02 --seed 1
launch 4 L4_cap5_anneal    --pool-cap 5  --p-anchor 0.5 --kl-target 0.05 --kl-target-final 0.30 --kl-anneal-updates 400 --entcoef 0.001 --seed 1
launch 5 L5_cap20_esc      --pool-cap 20 --p-anchor 0.2 --kl-target 0.05 --kl-target-final 0.50 --kl-anneal-updates 400 --entcoef 0.02 --seed 1
launch 6 L6_cap10_seed2    --pool-cap 10 --p-anchor 0.3 --kl-target 0.05 --kl-target-final 0.30 --kl-anneal-updates 400 --entcoef 0.001 --seed 2
launch 7 L7_cap30_lowanch  --pool-cap 30 --p-anchor 0.1 --kl-target 0.05 --kl-target-final 0.40 --kl-anneal-updates 400 --entcoef 0.001 --seed 1
echo LEAGUE_LAUNCHED
