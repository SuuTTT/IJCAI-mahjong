#!/bin/bash
# Autonomous overnight dev/train. Sequential (each step blocks), no foreground sleeps.
# Order: highest-value first (curriculum-v2), then the deeper-SL experiment + its distill.
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LOG=/tmp/overnight.log
echo "=== overnight START $(date) ===" > $LOG

echo "[1/3] curriculum-v2 from sl2distill (LONGER random stage to fix v1 overfit) $(date)" >> $LOG
CL_STATES=/tmp/curriculum_states.pkl python3 -u rl_curriculum.py \
  --base arch_ck/explore/resbn40_sl2distill.pkl --states /tmp/curriculum_states.pkl --blocks 40 \
  --stage-iters 30 --actors 18 --games-per-actor 4 --gauntlet-games 8 --beta-kl 0.6 \
  --out arch_ck/explore/resbn40_clv2.pkl >> $LOG 2>&1
echo "[1/3] curriculum-v2 done $(date)" >> $LOG

echo "[2/3] deeper SL: resbn56 from scratch (does more capacity beat sl2 val_acc 0.887?) $(date)" >> $LOG
python3 -u supervised_v2.py --warm '' --blocks 56 --epochs 10 --bs 768 --lr 3e-4 --aug 0.8 \
  --out arch_ck/explore/resbn56_sl2.pkl >> $LOG 2>&1
echo "[2/3] resbn56 SL done $(date)" >> $LOG

echo "[3/3] distill champion on resbn56 base $(date)" >> $LOG
python3 -u distill.py finetune_frac --base arch_ck/explore/resbn56_sl2.pkl --champ /tmp/champ_all.npz \
  --kind resbn --cfg '{"channels":128,"blocks":56}' --champ-frac 0.3 --steps 2800 --lr 5e-5 \
  --out arch_ck/explore/resbn56_sl2distill.pkl >> $LOG 2>&1
echo "[3/3] resbn56 distill done $(date)" >> $LOG
echo "=== overnight DONE $(date) ===" >> $LOG
