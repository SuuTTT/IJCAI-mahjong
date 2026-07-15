#!/bin/bash
# DETERMINISTIC-target track, on GPUs 4-7. Waits for the 8 det teachers (already running on 4-7)
# to finish, then 3 KD students, then the det 3-way gate.
cd /root/crossgame/doudizhu || exit 1
mkdir -p ckpt/students_det logs results
log(){ echo "[det $(date +%F_%T)] $*"; }

DET8="ckpt/teachers/dou_teacher_s0.pkl,ckpt/teachers/dou_teacher_s1.pkl,ckpt/teachers/dou_teacher_s2.pkl,ckpt/teachers/dou_teacher_s3.pkl,ckpt/teachers/dou_teacher_s4.pkl,ckpt/teachers/dou_teacher_s5.pkl,ckpt/teachers/dou_teacher_s6.pkl,ckpt/teachers/dou_teacher_s7.pkl"

# ---- wait for 8 det teachers to finish (pattern is specific to det-teacher --out) ----
log "waiting for 8 det teachers on 4-7 to finish"
until [ -f ckpt/teachers/dou_teacher_s7.pkl ] && [ "$(pgrep -cf 'out ckpt/teachers/dou_teacher_s')" = "0" ]; do sleep 20; done
log "det teachers finished"

# ---- 3 KD students seeds 100-102 on GPUs 4,5,6 ----
GPUS=(4 5 6); i=0
for sd in 100 101 102; do
  g=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$g nohup python3 dou_kd_train.py --data dou_data.npz --teachers "$DET8" \
      --seed $sd --steps 25000 --alpha 0.7 --out ckpt/students_det/dou_student_s${sd}.pkl \
      > logs/student_det_s${sd}.log 2>&1 &
  log "launched DET student seed=$sd GPU=$g pid=$!"; i=$((i+1)); sleep 15
done
until [ -f ckpt/students_det/dou_student_s102.pkl ] && [ "$(pgrep -cf 'out ckpt/students_det/dou_student_s')" = "0" ]; do sleep 20; done
log "det students finished"

# ---- det 3-way gate on GPU 7 (free during det-gate) ----
log "running deterministic 3-way gate on GPU 7"
CUDA_VISIBLE_DEVICES=7 python3 run_gate3.py --regime deterministic \
  --single ckpt/students_det/dou_student_s100.pkl \
  --teacher_ens ckpt/teachers/dou_teacher_s0.pkl,ckpt/teachers/dou_teacher_s1.pkl,ckpt/teachers/dou_teacher_s2.pkl \
  --student_ens ckpt/students_det/dou_student_s100.pkl,ckpt/students_det/dou_student_s101.pkl,ckpt/students_det/dou_student_s102.pkl \
  --seat 0 --nseeds 2000 --out results/det_gate.json > logs/gate_det.log 2>&1
log "DET GATE JSON WRITTEN -> results/det_gate.json"
log "DET TRACK COMPLETE"
