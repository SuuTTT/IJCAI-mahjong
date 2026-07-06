#!/bin/bash
cd /root/IJCAI-mahjong/train/caiest_repro
set -e
# wait for plain_bc to finish
while pgrep -f awr_bc.py >/dev/null; do sleep 10; done
MOYU=/root/assets/moyu_bn_128x40.pkl
SEEDS=800; W=100; S0=80000
GD=ckpt/awr/gates; mkdir -p $GD
# calib re-confirm at these seeds
python3 frontier_gate.py --cand $MOYU --cand-kind resbn --cand-cfg "channels=128,blocks=40" \
  --ref $MOYU --ref-kind resbn --ref-cfg "channels=128,blocks=40" \
  --knob off --seeds $SEEDS --workers $W --seed0 $S0 --out $GD/calib.json > $GD/calib.glog 2>&1
for v in plain_bc exp_b1 exp_b2 lin_k32; do
  CAND=ckpt/awr/moyu_${v}.pkl
  [ "$v" = "plain_bc" ] && CAND=ckpt/awr/moyu_plain_bc.pkl
  python3 frontier_gate.py --cand $CAND --cand-kind resbn_fused --cand-cfg "channels=128,blocks=40" \
    --ref $MOYU --ref-kind resbn --ref-cfg "channels=128,blocks=40" \
    --knob off --seeds $SEEDS --workers $W --seed0 $S0 --out $GD/$v.json > $GD/$v.glog 2>&1
  echo "GATE $v done: $(python3 -c "import json;d=json.load(open(\"$GD/$v.json\"));print(d[\"placement_pts\"],d[\"dist_pct\"])")"
done
echo "ALL_GATES_DONE"
