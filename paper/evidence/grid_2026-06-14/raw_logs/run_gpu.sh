#!/bin/bash
set -u; cd /root/mahjong
BETAS="${BETAS:-0.3 0.5 0.7 0.9 1.0 0.2}"
ALL="mythos_full typec_full chunjiandu_full resnet50_full damselfish_full scale_2000 scale_4000 scale_8000 scale_16000 "
mkdir -p res
for beta in $BETAS; do for t in $ALL; do
  out="cand_${beta}_${t}.pkl"
  [ -f "$out" ] && continue
  ( cd train/caiest_repro && python3 distill_kl.py --base ../../data/cnn_lad_chunjiandu.pkl \
      --champ ../../data/$t.npz --aw --beta "$beta" --steps 700 --out "/root/mahjong/$out" \
      > "/root/mahjong/res/distill_${beta}_${t}.log" 2>&1 )
  echo "$(date -u +%H:%M) distilled ${beta}_${t}" >> res/GPU.log
done; done
echo "GPU_LOOP_DONE $(date -u)" >> res/GPU.log
