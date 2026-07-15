#!/bin/bash
# 8-GPU F3-attack sweep: kdens3 warm-start + PopArt critic, B=256 bf16.
# Attacks F3 causes: (1) tight trust region -> widen KL; (3) SL basin -> entropy escalation.
# Each run self-logs strength-vs-kdens3-anchor to <out>/seed<seed>_jax_results.jsonl
# (built-in eval: seat0=current policy vs seats1-3=FROZEN kdens3; anchor-self baseline win_rate~0.233).
set -u
REPO=/root/ludus_rl
DUR=${DUR:-28800}          # 8h hard cap per run
COMMON="--B 256 --T 256 --dtype bf16 --popart 1 --value-epochs 6 \
  --updates 100000 --eval-every-updates 25 --eval-games 1024 --eval-at-start --log-every 1"
# gpu  seed  kl     ent     tag
CFG=(
 "0 1 0.05 0.001 control_kl05"
 "1 1 0.15 0.001 kl15"
 "2 1 0.30 0.001 kl30"
 "3 1 0.30 0.010 kl30_ent01"
 "4 1 0.30 0.030 kl30_ent03_escape"
 "5 1 0.50 0.050 kl50_ent05_noleash"
 "6 2 0.15 0.010 kl15_ent01_s2"
 "7 2 0.30 0.030 kl30_ent03_s2"
)
for row in "${CFG[@]}"; do
  read gpu seed kl ent tag <<< "$row"
  out=/root/rl_sweep/g${gpu}_${tag}
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu setsid bash -c \
    "cd $REPO && timeout $DUR python3 -m baselines.mahjong_t2_jax_v2 $COMMON \
       --seed $seed --kl-target $kl --entcoef $ent --out $out/ \
       > $out/run.log 2>&1" < /dev/null > /dev/null 2>&1 &
  echo "launched GPU$gpu seed=$seed kl=$kl ent=$ent -> $out"
  sleep 3
done
echo "all launched"
