#!/bin/bash
# Agent-owned noisy-H1 replication. Trains 8 students (seeds 7700-7707) distilled from ALL 8 noisy
# teachers into ckpt/students_repl_agent/ (dir I own), pools them with the existing all-8-distilled
# students (coordinator's 6000-6005 + 5000, read-only, + my 210-212), then runs the 6-block gap
# replication. Shares GPUs with the sweep via the same lock (no throttling of the sweep).
cd /root/crossgame/doudizhu || exit 1
mkdir -p ckpt/students_repl_agent logs results
log(){ echo "[replagent $(date +%F_%T)] $*"; }
LOCK=/tmp/sweep2_gpu.lock

pick_gpu(){  # echoes a free GPU (mem<2000 & <3 procs), lock-serialized
  local g="" used napps cand
  while ! mkdir "$LOCK" 2>/dev/null; do sleep 2; done
  while [ -z "$g" ]; do
    for cand in 0 1 2 3 4 5 6 7; do
      used=$(nvidia-smi -i $cand --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
      [ -z "$used" ] && continue
      napps=$(nvidia-smi -i $cand --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
      if [ "$used" -lt 2000 ] && [ "$napps" -lt 3 ]; then g=$cand; break; fi
    done
    [ -z "$g" ] && sleep 12
  done
  rmdir "$LOCK"; echo $g
}
launch_job(){ local lf="$1"; shift; local g=$(pick_gpu); CUDA_VISIBLE_DEVICES=$g nohup "$@" > "$lf" 2>&1 &
  log "launched on GPU=$g -> $lf"; sleep 22; }

TEA8="ckpt/teachers_noisy/dou_nteacher_s200.pkl,ckpt/teachers_noisy/dou_nteacher_s201.pkl,ckpt/teachers_noisy/dou_nteacher_s202.pkl,ckpt/teachers_noisy/dou_nteacher_s203.pkl,ckpt/teachers_noisy/dou_nteacher_s204.pkl,ckpt/teachers_noisy/dou_nteacher_s205.pkl,ckpt/teachers_noisy/dou_nteacher_s206.pkl,ckpt/teachers_noisy/dou_nteacher_s207.pkl"

log "training 8 agent students (7700-7707) distilled from all 8 noisy teachers"
for sd in 7700 7701 7702 7703 7704 7705 7706 7707; do
  launch_job logs/astudent_s${sd}.log python3 dou_kd_train.py --data dou_data_noisy.npz \
      --teachers "$TEA8" --seed $sd --steps 25000 --alpha 0.7 \
      --out ckpt/students_repl_agent/dou_astudent_s${sd}.pkl
done
until [ "$(ls ckpt/students_repl_agent/dou_astudent_s*.pkl 2>/dev/null | wc -l)" -ge 8 ] && [ "$(pgrep -cf 'out ckpt/students_repl_agent/dou_astudent')" = "0" ]; do sleep 20; done
log "8 agent students done"

# build pool from existing all-8-distilled students (only ones that exist)
POOL=""
for f in \
  ckpt/students_repl_agent/dou_astudent_s7700.pkl ckpt/students_repl_agent/dou_astudent_s7701.pkl \
  ckpt/students_repl_agent/dou_astudent_s7702.pkl ckpt/students_repl_agent/dou_astudent_s7703.pkl \
  ckpt/students_repl_agent/dou_astudent_s7704.pkl ckpt/students_repl_agent/dou_astudent_s7705.pkl \
  ckpt/students_repl_agent/dou_astudent_s7706.pkl ckpt/students_repl_agent/dou_astudent_s7707.pkl \
  ckpt/students_repl2/dou_r2student_s6000.pkl ckpt/students_repl2/dou_r2student_s6001.pkl \
  ckpt/students_repl2/dou_r2student_s6002.pkl ckpt/students_repl2/dou_r2student_s6003.pkl \
  ckpt/students_repl2/dou_r2student_s6004.pkl ckpt/students_repl2/dou_r2student_s6005.pkl \
  ckpt/students_repl/dou_rstudent_s5000.pkl \
  ckpt/students_noisy/dou_nstudent_s210.pkl ckpt/students_noisy/dou_nstudent_s211.pkl \
  ckpt/students_noisy/dou_nstudent_s212.pkl ; do
  [ -f "$f" ] && POOL="$POOL,$f"
done
POOL=${POOL#,}
np=$(echo "$POOL" | tr ',' '\n' | grep -c .)
log "pool size = $np students"

gg=$(pick_gpu)
log "running repl_h1 on GPU=$gg (pool=$np -> $((np/3)) blocks)"
CUDA_VISIBLE_DEVICES=$gg python3 repl_h1.py --students "$POOL" --teachers "$TEA8" \
    --nseeds 2000 --seed0 30000 --group 3 --out results/noisy_h1_repl.json > logs/repl_h1.log 2>&1
log "REPL H1 DONE -> results/noisy_h1_repl.json"
