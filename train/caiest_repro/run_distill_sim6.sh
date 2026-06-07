#!/bin/bash
# Driver: wait for preprocess_chunked.py to finish, then re-distill distill100b on the
# sim-6 diverse top-player data. OOM fallback: halve batch, double steps.
cd /root/mahjong/caiest_repro
LOG=/root/mahjong/distill_sim6.log
echo "=== driver START $(date) ===" >> $LOG

# preprocess_chunked prints 'done: cooked_{obs,mask,act}.npy' as its last line
while ! grep -q '^done: cooked_' /root/mahjong/preprocess.log 2>/dev/null; do
  # bail out if preprocess died without finishing
  if ! pgrep -f preprocess_chunked.py > /dev/null && ! grep -q '^done: cooked_' /root/mahjong/preprocess.log 2>/dev/null; then
    echo "PREPROCESS DIED $(date)" >> $LOG; exit 1
  fi
  sleep 60
done
echo "preprocess done, starting distill $(date)" >> $LOG

ARGS="finetune_frac --base /root/mahjong/ckpt/distill100b_fused.pkl \
  --champ data/topwinners_plus.npz --kind resbn_fused --cfg {\"channels\":128,\"blocks\":40} \
  --champ-frac 0.3 --lr 5e-5 --out /root/mahjong/ckpt/distill_sim6_fused.pkl"

if python3 -u distill.py $ARGS --steps 2800 --bs 1024 >> $LOG 2>&1; then
  echo "=== distill DONE bs1024 $(date) ===" >> $LOG
elif python3 -u distill.py $ARGS --steps 5600 --bs 512 >> $LOG 2>&1; then
  echo "=== distill DONE bs512-fallback $(date) ===" >> $LOG
else
  echo "=== distill FAILED both batch sizes $(date) ===" >> $LOG; exit 1
fi
