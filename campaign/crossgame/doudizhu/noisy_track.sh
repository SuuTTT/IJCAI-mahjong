#!/bin/bash
# NOISY-target track, on GPUs 0-3 (packing up to 2/GPU). Mahjong-aware: never colocates on a
# GPU that still runs an e13_kd_train/e12_ens_gate proc — waits for it to exit.
cd /root/crossgame/doudizhu || exit 1
mkdir -p ckpt/teachers_noisy ckpt/students_noisy logs results
log(){ echo "[noisy $(date +%F_%T)] $*"; }

declare -A MINE
for g in 0 1 2 3; do MINE[$g]=0; done

gpu_has_mahjong(){  # $1=gpu index -> 0 if a mahjong proc is on it
  local g=$1 pid
  for pid in $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do
    if tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -qE 'e13_kd_train|e12_ens_gate|caiest_repro'; then
      return 0; fi
  done
  return 1
}
pick_gpu(){  # echo a usable gpu in 0-3 (<2 of my jobs, no mahjong), else empty
  local g
  for g in 0 1 2 3; do
    if [ ${MINE[$g]} -lt 2 ] && ! gpu_has_mahjong $g; then echo $g; return; fi
  done
  echo ""
}

# ---- launch 6 noisy teachers ----
for sd in 200 201 202 203 204 205; do
  g=""
  while [ -z "$g" ]; do g=$(pick_gpu); [ -z "$g" ] && { log "no free 0-3 GPU (mahjong busy), waiting"; sleep 20; }; done
  CUDA_VISIBLE_DEVICES=$g nohup python3 dou_bc_train.py --data dou_data_noisy.npz \
      --seed $sd --steps 60000 --out ckpt/teachers_noisy/dou_nteacher_s${sd}.pkl \
      > logs/nteacher_s${sd}.log 2>&1 &
  MINE[$g]=$(( ${MINE[$g]} + 1 ))
  log "placed NOISY teacher seed=$sd on GPU=$g pid=$! (gpu now holds ${MINE[$g]} of my jobs)"
  sleep 30
done
log "ALL 6 NOISY TEACHERS PLACED"

# ---- wait for the 6 noisy teachers to finish ----
until [ -f ckpt/teachers_noisy/dou_nteacher_s205.pkl ] && [ "$(pgrep -cf 'out ckpt/teachers_noisy/dou_nteacher_s')" = "0" ]; do sleep 20; done
log "noisy teachers finished"
for g in 0 1 2 3; do MINE[$g]=0; done

# ---- 3 noisy KD students seeds 210-212 (on 0-3) ----
NTEA="ckpt/teachers_noisy/dou_nteacher_s200.pkl,ckpt/teachers_noisy/dou_nteacher_s201.pkl,ckpt/teachers_noisy/dou_nteacher_s202.pkl,ckpt/teachers_noisy/dou_nteacher_s203.pkl,ckpt/teachers_noisy/dou_nteacher_s204.pkl,ckpt/teachers_noisy/dou_nteacher_s205.pkl"
for sd in 210 211 212; do
  g=""; while [ -z "$g" ]; do g=$(pick_gpu); [ -z "$g" ] && sleep 15; done
  CUDA_VISIBLE_DEVICES=$g nohup python3 dou_kd_train.py --data dou_data_noisy.npz --teachers "$NTEA" \
      --seed $sd --steps 25000 --alpha 0.7 --out ckpt/students_noisy/dou_nstudent_s${sd}.pkl \
      > logs/nstudent_s${sd}.log 2>&1 &
  MINE[$g]=$(( ${MINE[$g]} + 1 ))
  log "placed NOISY student seed=$sd on GPU=$g pid=$!"
  sleep 20
done
until [ -f ckpt/students_noisy/dou_nstudent_s212.pkl ] && [ "$(pgrep -cf 'out ckpt/students_noisy/dou_nstudent_s')" = "0" ]; do sleep 20; done
log "noisy students finished"

# ---- noisy 3-way gate ----
gg=""; while [ -z "$gg" ]; do gg=$(pick_gpu); [ -z "$gg" ] && sleep 15; done
log "PHASE noisy-gate on GPU=$gg"
CUDA_VISIBLE_DEVICES=$gg python3 run_gate3.py --regime noisy_eps0.3 \
  --single ckpt/students_noisy/dou_nstudent_s210.pkl \
  --teacher_ens ckpt/teachers_noisy/dou_nteacher_s200.pkl,ckpt/teachers_noisy/dou_nteacher_s201.pkl,ckpt/teachers_noisy/dou_nteacher_s202.pkl \
  --student_ens ckpt/students_noisy/dou_nstudent_s210.pkl,ckpt/students_noisy/dou_nstudent_s211.pkl,ckpt/students_noisy/dou_nstudent_s212.pkl \
  --seat 0 --nseeds 2000 --out results/noisy_gate.json > logs/gate_noisy.log 2>&1
log "NOISY GATE JSON WRITTEN -> results/noisy_gate.json"
log "NOISY TRACK COMPLETE"
