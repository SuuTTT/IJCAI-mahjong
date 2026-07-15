#!/bin/bash
# LEAGUE relaunch (OOM fix): fixed KL (no anneal recompile) + B=128 (half VRAM). Pool = the lever.
# GPU2 keeps the running L2_cap10_klfix10 (u300+); relaunch the other 7.
TR=/root/ludus_rl/baselines/mahjong_t2_jax_v3_league.py
COMMON="--B 128 --T 256 --dtype bf16 --updates 100000 --pool-refresh 25"
# kill any stragglers from crashed configs (NOT L2)
for n in L0_cap10_anneal L1_cap20_anneal L3_cap10_esc L4_cap5_anneal L5_cap20_esc L6_cap10_seed2 L7_cap30_lowanch; do
  pkill -9 -f "rl_league/$n/" 2>/dev/null
done
sleep 4
launch(){ local g=$1 name=$2; shift 2
  setsid bash -c "CUDA_VISIBLE_DEVICES=$g timeout 25200 python3 $TR $COMMON $* --out /root/rl_league/$name/ > /root/rl_league/$name.log 2>&1" < /dev/null > /dev/null 2>&1 & }
launch 0 L0b_cap10        --pool-cap 10 --p-anchor 0.3 --kl-target 0.05 --entcoef 0.001 --seed 1
launch 1 L1b_cap20        --pool-cap 20 --p-anchor 0.3 --kl-target 0.05 --entcoef 0.001 --seed 1
launch 3 L3b_cap10_ent    --pool-cap 10 --p-anchor 0.3 --kl-target 0.05 --entcoef 0.02  --seed 1
launch 4 L4b_cap5_anch5   --pool-cap 5  --p-anchor 0.5 --kl-target 0.05 --entcoef 0.001 --seed 1
launch 5 L5b_cap20_kl10   --pool-cap 20 --p-anchor 0.2 --kl-target 0.10 --entcoef 0.02  --seed 1
launch 6 L6b_cap10_seed2  --pool-cap 10 --p-anchor 0.3 --kl-target 0.05 --entcoef 0.001 --seed 2
launch 7 L7b_cap15_lowanch --pool-cap 15 --p-anchor 0.1 --kl-target 0.05 --entcoef 0.001 --seed 1
echo LEAGUE2_LAUNCHED
