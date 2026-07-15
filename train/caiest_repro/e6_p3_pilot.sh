#!/bin/bash
# e6_p3_pilot.sh — pilot cross-hand estimator: 300 matches/field (prefix of the
# full 20M+ training seed block), curve only. Full run supersedes this.
set -x
cd /root/caiest_repro || exit 1
mkdir -p ckpt/e6_pilot logs data results
nice -n 10 python3 e6_match_data.py --nmatches 300 --hands 8 --workers 76 \
    --seed0 20000000 --out data/e6_match_pilot.npz > logs/e6p3_pilot_data.log 2>&1 \
    || { echo FAIL_PILOT_DATA; exit 1; }
nice -n 10 python3 e6_match_train.py --data data/e6_match_pilot.npz \
    --out results/E6_P3_PILOT_EST.json --ckptdir ckpt/e6_pilot > logs/e6p3_pilot_train.log 2>&1 \
    || { echo FAIL_PILOT_TRAIN; exit 1; }
echo PILOT_DONE
