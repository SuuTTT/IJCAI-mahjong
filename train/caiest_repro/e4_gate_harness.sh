#!/bin/bash
# E4 gate harness: duplicate-format placement gate for base calib + each AWR output,
# across 3 independent seed-blocks (for CIs). All vs moyu reference (128x40).
# Calib (moyu-vs-moyu) per block confirms ~2.500 baseline.
cd /root/IJCAI-mahjong/train/caiest_repro
MOYU=/root/assets/moyu_bn_128x40.pkl
SEEDS=500; W=60
GD=ckpt/e4/gates; mkdir -p $GD
SEED0S="70000 80000 90000"

# wait for training to finish (TRAIN_DONE flag) before gating
echo "$(date) gate harness: waiting for TRAIN_DONE" | tee -a logs/e4/gate_driver.log
while [ ! -f logs/e4/TRAIN_DONE ]; do
  [ -f /root/STOP_E4 ] && { echo "STOP_E4 — abort gates"; exit 1; }
  sleep 20
done
echo "$(date) TRAIN_DONE seen — begin gating" | tee -a logs/e4/gate_driver.log

gate() {  # blockidx seed0 name candpath candkind candcfg
  local bi=$1 s0=$2 nm=$3 cp=$4 ck=$5 cc=$6
  [ -f /root/STOP_E4 ] && return
  python3 frontier_gate.py --cand "$cp" --cand-kind "$ck" --cand-cfg "$cc" \
    --ref $MOYU --ref-kind resbn --ref-cfg "channels=128,blocks=40" \
    --knob off --seeds $SEEDS --workers $W --seed0 $s0 \
    --out $GD/${nm}_blk${bi}.json > $GD/${nm}_blk${bi}.glog 2>&1
  echo "$(date) GATE ${nm} blk${bi}(s0=$s0): $(python3 -c "import json;d=json.load(open(\"$GD/${nm}_blk${bi}.json\"));print(\"pts\",d[\"placement_pts\"])" 2>/dev/null)" | tee -a logs/e4/gate_driver.log
}

bi=0
for s0 in $SEED0S; do
  # calib block: moyu (raw resbn) vs moyu
  gate $bi $s0 calib $MOYU resbn "channels=128,blocks=40"
  # each AWR output
  for f in ckpt/e4/awr_b*.pkl; do
    nm=$(basename $f .pkl)
    gate $bi $s0 $nm $f resbn_fused "channels=128,blocks=40"
  done
  bi=$((bi+1))
done
echo "E4 gates ALL_DONE $(date)" | tee -a logs/e4/gate_driver.log
touch logs/e4/GATE_DONE
