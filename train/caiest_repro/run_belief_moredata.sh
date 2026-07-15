#!/bin/bash
cd /root/caiest_repro
# 1) generate 30k MORE self-play games (belief is data-limited: more data -> higher AUROC)
python3 oppbelief_gen.py --tag full2 --games 30000 --seed0 6000000 --workers 55 >> logs/belief_moregen.log 2>&1
# 2) merge full2 shards into full/ with offset names (distinct seeds -> disjoint games)
N=$(ls data/oppbelief/full/*.npz 2>/dev/null | wc -l); i=0
for f in data/oppbelief/full2/shard_*.npz; do
  ln -sf "$(realpath "$f")" "data/oppbelief/full/shard_$(printf %04d $((N+i))).npz"; i=$((i+1))
done
echo "merged: full now $(ls data/oppbelief/full/*.npz | wc -l) shards (~60k games)" >> logs/belief_moregen.log
# 3) retrain 3 seeds on doubled data, GPUs 5,6,7
mkdir -p ckpt/oppbelief
for pair in "5 20" "6 21" "7 22"; do set -- $pair
  CUDA_VISIBLE_DEVICES=$1 python3 oppbelief_train.py --tag full --seed $2 --steps 50000 --out ckpt/oppbelief/oppbelief_more60k_s$2.pt >> logs/belief_more60k_g$1.log 2>&1 &
done
wait
touch results/BELIEF_MORE60K_DONE
