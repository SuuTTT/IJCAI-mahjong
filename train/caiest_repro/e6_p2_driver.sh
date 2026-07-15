#!/bin/bash
# e6_p2_driver.sh — E6 Phase 2 pipeline: estimator data -> train -> switcher eval -> e12 gate.
# Run under setsid: setsid nohup bash e6_p2_driver.sh > logs/e6_p2_driver.log 2>&1 &
set -x
cd /root/caiest_repro || exit 1
mkdir -p results/e6_gate logs data ckpt/e6

echo "=== E6-P2 driver start $(date -u +%FT%TZ)"

# (a) estimator training data: kdens3 vs the 4 Phase-1 fields, fresh seeds 10M+
python3 e6_estimator_data.py --ngames 2500 --workers 110 --seed0 10000000 \
    --out data/e6_est_snaps.npz > logs/e6_data.log 2>&1 || { echo FAIL_DATA; exit 1; }

# (a) per-turn classifiers + chosen_T
python3 e6_estimator_train.py --data data/e6_est_snaps.npz \
    --out results/E6_ESTIMATOR.json --ckptdir ckpt/e6 > logs/e6_train.log 2>&1 \
    || { echo FAIL_TRAIN; exit 1; }

# (b) switcher on the 4 Phase-1 fields, Phase-1 kdens3 seeds (paired)
python3 e6_switcher_eval.py --ngames 2000 --workers 110 \
    --out results/E6_SWITCHER.json > logs/e6_switch.log 2>&1 || { echo FAIL_SWITCH; exit 1; }

# (c) standard e12 duplicate gate vs plain kdens3: 12 blocks x 500 seeds x 4 rot
for b in $(seq 0 11); do
  python3 e6_switcher_gate.py --seeds 500 --workers 110 --seed0 $((8000000 + b*500)) \
      --out results/e6_gate/e6sw_b${b}.json > logs/e6_gate_b${b}.log 2>&1 \
      || { echo FAIL_GATE_B${b}; exit 1; }
done
python3 e6_switcher_gate.py --agg 'results/e6_gate/e6sw_b*.json' \
    --aggout results/E6_SWITCHER_GATE.json > logs/e6_gate_agg.log 2>&1 \
    || { echo FAIL_GATE_AGG; exit 1; }

touch results/E6_P2_DONE
echo "=== E6-P2 driver done $(date -u +%FT%TZ)"
