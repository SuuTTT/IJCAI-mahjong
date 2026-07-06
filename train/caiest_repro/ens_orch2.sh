#!/bin/bash
cd /root/IJCAI-mahjong/train/caiest_repro
A=ckpt/aug/aug_128x40_s0.pkl; B=ckpt/aug/aug_128x40_s1.pkl; C=ckpt/aug/aug_128x40_s2.pkl
D=ckpt/e1b/full_128x40_s1.pkl
G192=/root/realfield_build/big192x40_s0_fused.pkl@192
# phase 1: extend ens4 to 24 blocks (higher power on the borderline tie)
for i in $(seq 12 23); do
  S0=$((210000 + i*1000))
  while [ "$(awk "{print (\$1>105)?1:0}" /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f ens_blocks/ens4_b$i.json ] || python3 e12_ens_gate.py --cand $A,$B,$C,$D --ref $A --seeds 500 --workers 64 --seed0 $S0 --out ens_blocks/ens4_b$i.json
done
# phase 2: DIVERSE ensemble (seeds + big192) 12 blocks
for i in $(seq 0 11); do
  S0=$((240000 + i*1000))
  while [ "$(awk "{print (\$1>105)?1:0}" /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f ens_blocks/ensdiv_b$i.json ] || python3 e12_ens_gate.py --cand $A,$B,$C,$G192 --ref $A --seeds 500 --workers 64 --seed0 $S0 --out ens_blocks/ensdiv_b$i.json
done
python3 - << "PYEOF"
import json, glob, numpy as np
T={11:2.201,12:2.179,23:2.069}
out={"experiment":"ENSEMBLE round 2: ens4 24-block extension + diverse (3 aug seeds + big192)","variants":{}}
for tag in ["ens3","ens4","ensdiv"]:
    fs=sorted(glob.glob(f"ens_blocks/{tag}_b*.json"))
    if not fs: continue
    v=np.array([json.load(open(f))["placement_pts"] for f in fs]); n=len(v)
    se=v.std(ddof=1)/np.sqrt(n); tc=T.get(n-1,2.06); lo=v.mean()-tc*se
    out["variants"][tag]=dict(n_blocks=n,games=n*2000,mean=round(float(v.mean()),4),
      sd=round(float(v.std(ddof=1)),4),ci95_lo=round(float(lo),4),margin_lo=round(float(lo-2.5),4),
      blocks=[round(float(x),4) for x in v],
      verdict="BEATS_AUGS0" if lo>2.5 else "TIED_NOT_SEPARATED")
json.dump(out,open("ENSEMBLE_R2_RESULTS.json","w"),indent=2)
print(json.dumps({k:{kk:vv for kk,vv in x.items() if kk!="blocks"} for k,x in out["variants"].items()},indent=2))
PYEOF
echo "ENS ROUND2 DONE"
