#!/bin/bash
# CHAIN: cook done -> GPU3 free -> retrain value (in-dist post states) -> E14 fixed value-guided sweep
cd /root/IJCAI-mahjong/train/caiest_repro
while [ ! -f data/cooked_value_post.npz ]; do sleep 120; done
grep -q "wrote" logs/cook_post.log || sleep 60
while pgrep -f rl_paired.py > /dev/null; do sleep 120; done
CUDA_VISIBLE_DEVICES=3 python3 train_value_multi.py \
  --data data/cooked_value.npz,data/cooked_value_post.npz \
  --channels 256 --blocks 40 --epochs 6 --gpu 0 \
  --out ckpt/value_post_256x40.pkl --json_out VALUE_POST_TRAIN.json > logs/value_post_train.log 2>&1
[ -f ckpt/value_post_256x40.pkl ] || { echo "VALUE TRAIN FAILED" >> value_chain.log; exit 1; }
A=ckpt/aug/aug_128x40_s0.pkl
mkdir -p e14_blocks
# calibration: lam=0 must be exactly 2.500
python3 e14_gate.py --cand $A --ref $A --value ckpt/value_post_256x40.pkl --lam 0 \
  --seeds 250 --workers 60 --seed0 300000 --out e14_blocks/calib.json
for LAM in 0.25 0.5 1.0; do
  for i in $(seq 0 5); do
    S=$((310000 + i*1000))
    while [ "$(awk "{print (\$1>105)?1:0}" /proc/loadavg)" = "1" ]; do sleep 60; done
    [ -f e14_blocks/lam${LAM}_b$i.json ] || python3 e14_gate.py --cand $A --ref $A \
      --value ckpt/value_post_256x40.pkl --lam $LAM --topk 5 \
      --seeds 500 --workers 60 --seed0 $S --out e14_blocks/lam${LAM}_b$i.json
  done
done
python3 - << "PYEOF"
import json, glob, numpy as np, sys
out = {"experiment": "E14 FIXED value-guided (post-discard obs rendered; in-dist value net)",
       "note": "E8 was a silent no-op on discards (request2obs None for self-Play)", "lams": {}}
c = json.load(open("e14_blocks/calib.json")); out["calibration"] = {"placement_pts": c["placement_pts"], "hooked_rate": c.get("hooked_rate")}
for lam in ["0.25", "0.5", "1.0"]:
    fs = sorted(glob.glob(f"e14_blocks/lam{lam}_b*.json"))
    if not fs: continue
    js = [json.load(open(f)) for f in fs]
    v = np.array([j["placement_pts"] for j in js]); n = len(v)
    se = v.std(ddof=1)/np.sqrt(n); lo = v.mean()-{5:2.571,11:2.201}.get(n-1,2.3)*se
    out["lams"][lam] = dict(n_blocks=n, mean=round(float(v.mean()),4), ci95_lo=round(float(lo),4),
        hooked_rate=js[0].get("hooked_rate"), override_rate=js[0].get("override_rate"),
        blocks=[round(float(x),4) for x in v],
        verdict="BEATS_AUGS0" if lo>2.5 else "TIED_NOT_SEPARATED")
# LOUDFAIL checks: no silent partial results
problems = []
if abs(out["calibration"]["placement_pts"] - 2.5) > 1e-9: problems.append("CALIBRATION NOT 2.500")
for lam in ["0.25", "0.5", "1.0"]:
    got = out["lams"].get(lam, {}).get("n_blocks", 0)
    if got < 6: problems.append("lam %s: %d/6 blocks MISSING" % (lam, got))
    hr = out["lams"].get(lam, {}).get("hooked_rate") or 0
    if hr < 0.9: problems.append("lam %s: hooked_rate %s < 0.9 (mechanism not engaged)" % (lam, hr))
out["integrity"] = "OK" if not problems else problems
json.dump(out, open("E14_RESULTS.json","w"), indent=2)
print(json.dumps(out, indent=2))
if problems:
    print("E14 AGGREGATION INTEGRITY FAIL:", problems); sys.exit(2)
PYEOF
echo "E14 CHAIN DONE" >> value_chain.log
