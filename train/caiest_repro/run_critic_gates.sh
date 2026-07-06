#!/bin/bash
# Gate each critic-AWR candidate vs moyu reference. Calib (moyu vs moyu) must read 2.5.
# Uses the SAME seeds as gate_calib_moyu.json (seed0=70000, seeds=400) so numbers compare directly.
cd /root/IJCAI-mahjong/train/caiest_repro
MOYU=/root/assets/moyu_bn_128x40.pkl
SEEDS=400; W=100; S0=70000
GD=ckpt/rl/gates; mkdir -p $GD

gate() {  # name candpath candcfg
  python3 frontier_gate.py --cand "$2" --cand-kind resbn_fused --cand-cfg "$3" \
    --ref $MOYU --ref-kind resbn --ref-cfg "channels=128,blocks=40" \
    --knob off --seeds $SEEDS --workers $W --seed0 $S0 --out $GD/$1.json > $GD/$1.glog 2>&1
  echo "GATE $1: $(python3 -c "import json;d=json.load(open('$GD/$1.json'));print('pts',d['placement_pts'],'dist%',d['dist_pct'],'1st%',d['first_pct'],'4th%',d['fourth_pct'])")"
}

# calib: moyu vs moyu (raw resbn, not fused) -> expect 2.5
python3 frontier_gate.py --cand $MOYU --cand-kind resbn --cand-cfg "channels=128,blocks=40" \
  --ref $MOYU --ref-kind resbn --ref-cfg "channels=128,blocks=40" \
  --knob off --seeds $SEEDS --workers $W --seed0 $S0 --out $GD/calib.json > $GD/calib.glog 2>&1
echo "CALIB: $(python3 -c "import json;d=json.load(open('$GD/calib.json'));print('pts',d['placement_pts'])")"

for v in b05 b1 b2; do
  gate critic_$v ckpt/rl/moyu_critic_$v.pkl "channels=128,blocks=40"
done
echo "ALL_128_GATES_DONE"

# 256-actor variant: gate its BASE (big256x40) and its AWR version, both vs moyu reference.
# (Base != moyu so its baseline isn't 2.5; the AWR-vs-base delta tells us if AWR moved it.)
gate big256_base   ckpt/big256x40_s0_fused.pkl  "channels=256,blocks=40"
gate big256_critic ckpt/rl/big256_critic_b1.pkl "channels=256,blocks=40"
echo "ALL_GATES_DONE"
