#!/bin/bash
# 2027 search leaf-evaluator: value-head ensemble (seeds 1,2 join VALUE_C_60K seed 0).
cd /root/caiest_repro
for s in 1 2; do
  out=results/VALUE_C_60K_s$s.json
  [ -f "$out" ] && continue
  CUDA_VISIBLE_DEVICES=1 python3 f2_value_v2.py --variant c --seed $s --steps 60000 --out $out >> logs/value_c_ens.log 2>&1
done
touch results/VALUE_ENS_DONE; echo VALUE_ENS_DONE
