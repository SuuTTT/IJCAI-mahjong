#!/bin/bash
# 2027 candidate placement gates (paired duplicate, 12 disjoint blocks/gate).
# cond models selected on held-out final2-val (NOT gate walls) -> walls are fresh.
cd /root/caiest_repro
CONDENS=ckpt/cond/cond_s0.pkl,ckpt/cond/cond_s1.pkl,ckpt/cond/cond_s2.pkl
ZEROENS=ckpt/cond/zero_s0.pkl,ckpt/cond/zero_s1.pkl,ckpt/cond/zero_s2.pkl
KDENS3=ckpt/kd/kd_128x40_s0.pkl,ckpt/kd/kd_128x40_s1.pkl,ckpt/kd/kd_128x40_s2.pkl
mkdir -p results/cond_gate

gate12 () {  # name, cand, cand_planes, cand_src, ref, ref_planes, ref_src_arg, workers
  local NAME=$1 CAND=$2 CP=$3 CS=$4 REF=$5 RP=$6 RSARG=$7 W=$8
  for b in $(seq 0 11); do
    local s0=$((7000000 + b*3000))   # disjoint blocks, fresh region, step>>500 games
    local out=results/cond_gate/${NAME}_b${b}.json
    [ -f "$out" ] && continue
    python3 e12_cond_gate.py --cand "$CAND" --cand_planes $CP --cand_src $CS \
      --ref "$REF" --ref_planes $RP $RSARG --seeds 500 --workers $W --seed0 $s0 --out "$out" \
      >> logs/cond_gate_${NAME}.log 2>&1
  done
}

# Gate 1 (the headline): conditioned-ens vs kdens3  -- does 2027 candidate beat kdens3?
gate12 cond_vs_kdens3 "$CONDENS" 39 1.0 "$KDENS3" 38 "" 46 &
# Gate 2: conditioned-ens(src=1) vs unconditioned-ens(src=0) -- conditioning effect on the METRIC
gate12 cond_vs_zero "$CONDENS" 39 1.0 "$ZEROENS" 39 "--ref_src 0.0" 46 &
wait
python3 - << PYEOF
import json, glob, math
def agg(name):
    vals=[json.load(open(f))["placement_pts"] for f in sorted(glob.glob(f"results/cond_gate/{name}_b*.json"))]
    if not vals: return None
    n=len(vals); m=sum(vals)/n
    sd=(sum((v-m)**2 for v in vals)/(n-1))**0.5 if n>1 else 0.0
    ci=1.96*sd/n**0.5
    return dict(name=name, blocks=n, mean=round(m,4), ci=round(ci,4),
               lo=round(m-ci,4), hi=round(m+ci,4), beats_2500=bool(m-ci>2.5), vals=[round(v,4) for v in vals])
out={k:agg(k) for k in ["cond_vs_kdens3","cond_vs_zero"]}
json.dump(out, open("results/COND_GATE.json","w"), indent=1)
print(json.dumps(out, indent=1))
PYEOF
touch results/COND_GATE_DONE; echo COND_GATE_DONE
