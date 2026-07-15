#!/bin/bash
cd /root/caiest_repro
T6=ckpt/aug/aug_128x40_s0.pkl,ckpt/aug/aug_128x40_s1.pkl,ckpt/aug/aug_128x40_s2.pkl,ckpt/aug/aug_128x40_s3.pkl,ckpt/aug/aug_128x40_s4.pkl,ckpt/aug/aug_128x40_s5.pkl
mkdir -p ckpt/seproper logs/seproper
free_gpu(){ for g in 0 1 2 3 4 5 6 7; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null); n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null|grep -c .); [ "${m:-99999}" -lt 3000 ]&&[ "${n:-9}" -lt 2 ]&&{ echo $g; return; }; done; echo -1; }
# wait for daemon trainers to drain so GPUs free up
while [ "$(pgrep -cf e13_kd_frac)" -gt 4 ]; do sleep 60; done
for frac in 0.02 0.05 0.10 0.25 0.50 1.00; do
  for s in 9000 9001 9002; do
    ftag=$(echo $frac|sed "s/\.//")
    out=ckpt/seproper/se_f${ftag}_s${s}.pkl
    [ -f "$out" ] && continue
    g=-1; while [ "$g" = "-1" ]; do g=$(free_gpu); [ "$g" = "-1" ]&&sleep 40; done
    CUDA_VISIBLE_DEVICES=$g nohup python3 e13_kd_frac.py --channels 128 --blocks 40 --steps 60000 --seed $s --teachers $T6 --alpha 0.7 --frac $frac --out $out > logs/seproper/se_f${ftag}_s${s}.log 2>&1 &
    sleep 25
  done
done
while [ "$(pgrep -cf "seproper/se_f")" -gt 0 ]; do sleep 60; done
# gate each fractions 3-student ensemble
A=ckpt/aug/aug_128x40_s0.pkl; mkdir -p seproper_gate
for frac in 0.02 0.05 0.10 0.25 0.50 1.00; do
  ftag=$(echo $frac|sed "s/\.//")
  C=ckpt/seproper/se_f${ftag}_s9000.pkl,ckpt/seproper/se_f${ftag}_s9001.pkl,ckpt/seproper/se_f${ftag}_s9002.pkl
  for i in $(seq 0 7); do
    [ -f seproper_gate/f${ftag}_b$i.json ] || python3 e12_ens_gate.py --cand $C --ref $A --seeds 500 --workers 80 --seed0 $((520000+i*1000)) --out seproper_gate/f${ftag}_b$i.json
  done
done
python3 - << PYEOF
import json,glob,numpy as np
out={}
for frac in ["002","005","010","025","050","100"]:
    fs=sorted(glob.glob("seproper_gate/f%s_b*.json"%frac))
    if not fs: continue
    v=np.array([json.load(open(x))["placement_pts"] for x in fs]); n=len(v); se=v.std(ddof=1)/np.sqrt(n); lo=v.mean()-2.365*se
    out[frac]=dict(n=n,placement=round(float(v.mean()),4),ci_lo=round(float(lo),4),beats=bool(lo>2.5))
json.dump(out,open("SAMPEFF_PLACEMENT_PROPER.json","w"),indent=2); print(json.dumps(out,indent=2))
PYEOF
echo SEPROPER_DONE
