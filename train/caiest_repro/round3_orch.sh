#!/bin/bash
cd /root/IJCAI-mahjong/train/caiest_repro
A=ckpt/aug/aug_128x40_s0.pkl; B=ckpt/aug/aug_128x40_s1.pkl; C=ckpt/aug/aug_128x40_s2.pkl
G192=ckpt/big192x40_s0_fused.pkl@192
RL=ckpt/rl_paired/snap_rlpair_01975.pkl
for i in $(seq 0 11); do
  S0=$((240000 + i*1000))
  while [ "$(awk "{print (\$1>105)?1:0}" /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f ens_blocks/ensdiv_b$i.json ] || python3 e12_ens_gate.py --cand $A,$B,$C,$G192 --ref $A --seeds 500 --workers 60 --seed0 $S0 --out ens_blocks/ensdiv_b$i.json
  [ -f ens_blocks/rlsnap_b$i.json ] || python3 e8_gate.py --cand $RL --ref $A --lam 0 --seeds 500 --workers 60 --seed0 $((260000 + i*1000)) --out ens_blocks/rlsnap_b$i.json
done
python3 - << "PYEOF"
import json, glob, numpy as np
T={11:2.201}
out={}
for tag in ["ensdiv","rlsnap"]:
    fs=sorted(glob.glob(f"ens_blocks/{tag}_b*.json"))
    if not fs: continue
    v=np.array([json.load(open(f))["placement_pts"] for f in fs]); n=len(v)
    se=v.std(ddof=1)/np.sqrt(n); lo=v.mean()-T.get(n-1,2.2)*se
    out[tag]=dict(n_blocks=n,mean=round(float(v.mean()),4),ci95_lo=round(float(lo),4),
        margin_lo=round(float(lo-2.5),4),blocks=[round(float(x),4) for x in v],
        verdict="BEATS_AUGS0" if lo>2.5 else "TIED_NOT_SEPARATED")
json.dump(out,open("ROUND3_RESULTS.json","w"),indent=2)
print(json.dumps(out,indent=2))
PYEOF
echo "ROUND3 DONE"
