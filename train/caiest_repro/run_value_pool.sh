#!/bin/bash
# Robust value-head pool for 2027 (search leaf evaluator AND defense-model core).
# ~1/3 of seeds collapse to constant predictor (r~0); train a pool, keep r>0.6.
cd /root/caiest_repro
run () { local g=$1 s=$2; local out=results/VALUE_C_60K_s$s.json
  [ -f "$out" ] && return
  CUDA_VISIBLE_DEVICES=$g python3 f2_value_v2.py --variant c --seed $s --steps 60000 --out $out >> logs/value_pool_g$g.log 2>&1; }
run 0 3 & run 1 4 & run 2 5 & run 3 6 & run 7 7 &
wait
python3 - << PYEOF
import json, glob
pool={}
for f in sorted(glob.glob("results/VALUE_C_60K*.json")):
    d=json.load(open(f)); r=d["metrics_final2"]["r_all"]
    tag=f.split("VALUE_C_60K")[1][:-5] or "_s0"
    pool[tag]=round(r,4)
good=[k for k,v in pool.items() if v>0.6]
json.dump({"pool_r":pool,"good_heads":good,"n_good":len(good)}, open("results/VALUE_POOL.json","w"), indent=1)
print("pool r:", pool); print("GOOD (r>0.6):", good)
PYEOF
touch results/VALUE_POOL_DONE; echo VALUE_POOL_DONE
