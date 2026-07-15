#!/bin/bash
# Corrected NOISY track. seeds 200-203 already run on GPU0/GPU1 (clean). This script
# (re)launches seeds 204,205 on the FREE GPU3 (GPU2 still runs a live mahjong e13 job and must
# NOT be shared), waits for all 6, then 3 KD students on GPUs 0,1,3, then the noisy 3-way gate.
# Placement is HARDCODED to 0,1,3 (never GPU2) because mahjong runs in a different PID namespace
# and cannot be auto-detected via /proc.
cd /root/crossgame/doudizhu || exit 1
mkdir -p ckpt/teachers_noisy ckpt/students_noisy logs results
log(){ echo "[noisy2 $(date +%F_%T)] $*"; }

# ---- (re)launch the two vacated teachers on GPU3 ----
for sd in 204 205; do
  CUDA_VISIBLE_DEVICES=3 nohup python3 dou_bc_train.py --data dou_data_noisy.npz \
      --seed $sd --steps 60000 --out ckpt/teachers_noisy/dou_nteacher_s${sd}.pkl \
      > logs/nteacher_s${sd}.log 2>&1 &
  log "relaunched NOISY teacher seed=$sd on GPU=3 pid=$!"; sleep 20
done

# ---- wait for all 6 noisy teachers to finish ----
until [ -f ckpt/teachers_noisy/dou_nteacher_s205.pkl ] && [ "$(pgrep -cf 'out ckpt/teachers_noisy/dou_nteacher_s')" = "0" ]; do sleep 20; done
log "noisy teachers finished"

# ---- 3 noisy KD students seeds 210-212 on GPUs 0,1,3 ----
NTEA="ckpt/teachers_noisy/dou_nteacher_s200.pkl,ckpt/teachers_noisy/dou_nteacher_s201.pkl,ckpt/teachers_noisy/dou_nteacher_s202.pkl,ckpt/teachers_noisy/dou_nteacher_s203.pkl,ckpt/teachers_noisy/dou_nteacher_s204.pkl,ckpt/teachers_noisy/dou_nteacher_s205.pkl"
SG=(0 1 3); i=0
for sd in 210 211 212; do
  g=${SG[$i]}
  CUDA_VISIBLE_DEVICES=$g nohup python3 dou_kd_train.py --data dou_data_noisy.npz --teachers "$NTEA" \
      --seed $sd --steps 25000 --alpha 0.7 --out ckpt/students_noisy/dou_nstudent_s${sd}.pkl \
      > logs/nstudent_s${sd}.log 2>&1 &
  log "launched NOISY student seed=$sd GPU=$g pid=$!"; i=$((i+1)); sleep 15
done
until [ -f ckpt/students_noisy/dou_nstudent_s212.pkl ] && [ "$(pgrep -cf 'out ckpt/students_noisy/dou_nstudent_s')" = "0" ]; do sleep 20; done
log "noisy students finished"

# ---- noisy 3-way gate on GPU0 ----
log "running noisy 3-way gate on GPU0"
CUDA_VISIBLE_DEVICES=0 python3 run_gate3.py --regime noisy_eps0.3 \
  --single ckpt/students_noisy/dou_nstudent_s210.pkl \
  --teacher_ens ckpt/teachers_noisy/dou_nteacher_s200.pkl,ckpt/teachers_noisy/dou_nteacher_s201.pkl,ckpt/teachers_noisy/dou_nteacher_s202.pkl \
  --student_ens ckpt/students_noisy/dou_nstudent_s210.pkl,ckpt/students_noisy/dou_nstudent_s211.pkl,ckpt/students_noisy/dou_nstudent_s212.pkl \
  --seat 0 --nseeds 2000 --out results/noisy_gate.json > logs/gate_noisy.log 2>&1
log "NOISY GATE JSON WRITTEN -> results/noisy_gate.json"
log "NOISY TRACK COMPLETE"
