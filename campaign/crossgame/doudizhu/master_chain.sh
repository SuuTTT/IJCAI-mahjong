#!/bin/bash
# Sequential distill-then-ensemble chain, ALL on GPUs 4-7 (mahjong owns 0-3, never touched).
# Phases run strictly one-at-a-time so per-phase pgrep waits are unambiguous.
cd /root/crossgame/doudizhu || exit 1
mkdir -p ckpt/students_det ckpt/teachers_noisy ckpt/students_noisy logs results
log(){ echo "[chain $(date +%F_%T)] $*"; }

DET8="ckpt/teachers/dou_teacher_s0.pkl,ckpt/teachers/dou_teacher_s1.pkl,ckpt/teachers/dou_teacher_s2.pkl,ckpt/teachers/dou_teacher_s3.pkl,ckpt/teachers/dou_teacher_s4.pkl,ckpt/teachers/dou_teacher_s5.pkl,ckpt/teachers/dou_teacher_s6.pkl,ckpt/teachers/dou_teacher_s7.pkl"

# ================= PHASE A: wait for 8 deterministic teachers to FINISH =================
log "PHASE A: waiting for 8 det teachers to finish"
until [ -f ckpt/teachers/dou_teacher_s7.pkl ] && [ "$(pgrep -cf 'python3 dou_bc_train.py')" = "0" ]; do sleep 20; done
log "det teachers finished"

# ================= PHASE B: 3 KD students (det data) seeds 100-102 =================
GPUS=(4 5 6); i=0
for sd in 100 101 102; do
  g=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$g nohup python3 dou_kd_train.py --data dou_data.npz --teachers "$DET8" \
      --seed $sd --steps 25000 --alpha 0.7 --out ckpt/students_det/dou_student_s${sd}.pkl \
      > logs/student_det_s${sd}.log 2>&1 &
  log "launched DET student seed=$sd GPU=$g pid=$!"; i=$((i+1)); sleep 15
done
log "PHASE B: waiting for 3 det students"
until [ -f ckpt/students_det/dou_student_s102.pkl ] && [ "$(pgrep -cf 'python3 dou_kd_train.py')" = "0" ]; do sleep 20; done
log "det students finished"

# ================= PHASE C: 3-way gate (deterministic regime) =================
log "PHASE C: running deterministic-regime 3-way gate"
CUDA_VISIBLE_DEVICES=4 python3 run_gate3.py --regime deterministic \
  --single ckpt/students_det/dou_student_s100.pkl \
  --teacher_ens ckpt/teachers/dou_teacher_s0.pkl,ckpt/teachers/dou_teacher_s1.pkl,ckpt/teachers/dou_teacher_s2.pkl \
  --student_ens ckpt/students_det/dou_student_s100.pkl,ckpt/students_det/dou_student_s101.pkl,ckpt/students_det/dou_student_s102.pkl \
  --seat 0 --nseeds 2000 --out results/det_gate.json > logs/gate_det.log 2>&1
log "DET GATE JSON WRITTEN -> results/det_gate.json"

# ================= PHASE D: 6 noisy teachers seeds 200-205 =================
NGPUS=(4 5 6 7 4 5); i=0
for sd in 200 201 202 203 204 205; do
  g=${NGPUS[$i]}
  CUDA_VISIBLE_DEVICES=$g nohup python3 dou_bc_train.py --data dou_data_noisy.npz \
      --seed $sd --steps 60000 --out ckpt/teachers_noisy/dou_nteacher_s${sd}.pkl \
      > logs/nteacher_s${sd}.log 2>&1 &
  log "launched NOISY teacher seed=$sd GPU=$g pid=$!"; i=$((i+1)); sleep 20
done
log "PHASE D: waiting for 6 noisy teachers"
until [ -f ckpt/teachers_noisy/dou_nteacher_s205.pkl ] && [ "$(pgrep -cf 'python3 dou_bc_train.py')" = "0" ]; do sleep 20; done
log "noisy teachers finished"

# ================= PHASE E: 3 KD students (noisy data) seeds 210-212 =================
NTEA="ckpt/teachers_noisy/dou_nteacher_s200.pkl,ckpt/teachers_noisy/dou_nteacher_s201.pkl,ckpt/teachers_noisy/dou_nteacher_s202.pkl,ckpt/teachers_noisy/dou_nteacher_s203.pkl,ckpt/teachers_noisy/dou_nteacher_s204.pkl,ckpt/teachers_noisy/dou_nteacher_s205.pkl"
GPUS=(4 5 6); i=0
for sd in 210 211 212; do
  g=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$g nohup python3 dou_kd_train.py --data dou_data_noisy.npz --teachers "$NTEA" \
      --seed $sd --steps 25000 --alpha 0.7 --out ckpt/students_noisy/dou_nstudent_s${sd}.pkl \
      > logs/nstudent_s${sd}.log 2>&1 &
  log "launched NOISY student seed=$sd GPU=$g pid=$!"; i=$((i+1)); sleep 15
done
log "PHASE E: waiting for 3 noisy students"
until [ -f ckpt/students_noisy/dou_nstudent_s212.pkl ] && [ "$(pgrep -cf 'python3 dou_kd_train.py')" = "0" ]; do sleep 20; done
log "noisy students finished"

# ================= PHASE F: 3-way gate (noisy regime) =================
log "PHASE F: running noisy-regime 3-way gate"
CUDA_VISIBLE_DEVICES=4 python3 run_gate3.py --regime noisy_eps0.3 \
  --single ckpt/students_noisy/dou_nstudent_s210.pkl \
  --teacher_ens ckpt/teachers_noisy/dou_nteacher_s200.pkl,ckpt/teachers_noisy/dou_nteacher_s201.pkl,ckpt/teachers_noisy/dou_nteacher_s202.pkl \
  --student_ens ckpt/students_noisy/dou_nstudent_s210.pkl,ckpt/students_noisy/dou_nstudent_s211.pkl,ckpt/students_noisy/dou_nstudent_s212.pkl \
  --seat 0 --nseeds 2000 --out results/noisy_gate.json > logs/gate_noisy.log 2>&1
log "NOISY GATE JSON WRITTEN -> results/noisy_gate.json"
log "MASTER CHAIN COMPLETE"
