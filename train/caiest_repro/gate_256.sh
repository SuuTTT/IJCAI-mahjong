#!/bin/bash
cd /root/IJCAI-mahjong/train/caiest_repro
MOYU=/root/assets/moyu_bn_128x40.pkl
SEEDS=400; W=48; S0=70000
GD=ckpt/rl/gates; mkdir -p $GD
gate() {
  python3 frontier_gate.py --cand "$2" --cand-kind resbn_fused --cand-cfg "$3" \
    --ref $MOYU --ref-kind resbn --ref-cfg "channels=128,blocks=40" \
    --knob off --seeds $SEEDS --workers $W --seed0 $S0 --out $GD/$1.json > $GD/$1.glog 2>&1
  echo "GATE $1: $(python3 -c "import json;d=json.load(open('$GD/$1.json'));print('pts',d['placement_pts'],'dist%',d['dist_pct'],'1st%',d['first_pct'],'4th%',d['fourth_pct'])")"
}
python3 frontier_gate.py --cand $MOYU --cand-kind resbn --cand-cfg "channels=128,blocks=40" \
  --ref $MOYU --ref-kind resbn --ref-cfg "channels=128,blocks=40" \
  --knob off --seeds $SEEDS --workers $W --seed0 $S0 --out $GD/calib.json > $GD/calib.glog 2>&1
echo "CALIB: $(python3 -c "import json;d=json.load(open('$GD/calib.json'));print('pts',d['placement_pts'])")"
gate big256_base   ckpt/big256x40_s0_fused.pkl  "channels=256,blocks=40"
gate big256_critic ckpt/rl/big256_critic_b1.pkl "channels=256,blocks=40"
echo "GATE_256_DONE"
