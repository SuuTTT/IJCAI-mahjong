#!/bin/bash
# e6_p3_driver.sh — E6 Phase 3 pipeline: match data -> estimator -> eval precompute -> assemble.
# Run under setsid: setsid nohup bash e6_p3_driver.sh > logs/e6_p3_driver.log 2>&1 &
set -x
cd /root/caiest_repro || exit 1
mkdir -p results logs data ckpt/e6
W=${W:-80}
export OMP_NUM_THREADS=32   # HistGB: avoid 128-thread OpenMP oversubscription

echo "=== E6-P3 driver start $(date -u +%FT%TZ)"

# 1) training matches: 2000/field x 8 hands, kdens3 seat-0, seeds 20M+
nice -n 10 python3 e6_match_data.py --nmatches 2000 --hands 8 --workers $W \
    --seed0 20000000 --out data/e6_match_train.npz > logs/e6p3_data.log 2>&1 \
    || { echo FAIL_P3_DATA; exit 1; }

# 2) cross-hand estimator curve (bal-acc by h) + per-h models
nice -n 10 python3 e6_match_train.py --data data/e6_match_train.npz \
    --out results/E6_CROSSHAND_EST.json --ckptdir ckpt/e6 > logs/e6p3_train.log 2>&1 \
    || { echo FAIL_P3_TRAIN; exit 1; }
echo "=== estimator curve ready $(date -u +%FT%TZ)"
python3 -c "import json; print(json.load(open('results/E6_CROSSHAND_EST.json'))['bal_acc_by_h'])"

# 3) eval matches precomputed under both seat-0 modes: 2000/field x 8 hands x {kd,aug}, seeds 30M+
nice -n 10 python3 e6_match_precompute.py --nmatches 2000 --hands 8 --workers $W \
    --seed0 30000000 --out data/e6_match_eval.npz > logs/e6p3_pre.log 2>&1 \
    || { echo FAIL_P3_PRE; exit 1; }

# 4) assemble arms + verdict
nice -n 10 python3 e6_match_assemble.py --evaldata data/e6_match_eval.npz \
    --est results/E6_CROSSHAND_EST.json --ckptdir ckpt/e6 \
    --out results/E6_CROSSHAND.json > logs/e6p3_asm.log 2>&1 \
    || { echo FAIL_P3_ASM; exit 1; }

touch results/E6_P3_DONE
echo "=== E6-P3 driver done $(date -u +%FT%TZ)"
