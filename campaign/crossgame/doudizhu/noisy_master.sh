#!/bin/bash
# NOISY-regime master: completes 8 teachers (200-207), runs H1 gate, then the N={1,2,4,8}
# teacher-count curve. GPU picking is MEMORY-BASED (mem<1500MB = free): robust to PID-namespace,
# auto-avoids the mahjong GPU (~3600MB) and any busy GPU (mine, det-track, or sibling).
cd /root/crossgame/doudizhu || exit 1
mkdir -p ckpt/teachers_noisy ckpt/students_noisy ckpt/students_curve logs results
log(){ echo "[noisy_master $(date +%F_%T)] $*"; }

launch_job(){  # $1=logfile ; rest=command. Blocks until a GPU has mem<1500MB, launches there.
  local lf="$1"; shift; local g="" used cand
  while [ -z "$g" ]; do
    for cand in 0 1 3 7 4 5 6 2; do
      used=$(nvidia-smi -i $cand --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
      [ -z "$used" ] && continue
      if [ "$used" -lt 1500 ]; then g=$cand; break; fi
    done
    [ -z "$g" ] && sleep 15
  done
  CUDA_VISIBLE_DEVICES=$g nohup "$@" > "$lf" 2>&1 &
  log "launched on GPU=$g -> $lf"
  sleep 30   # let CUDA ctx + data load so the next mem check sees this GPU busy
}

T_ALL="ckpt/teachers_noisy/dou_nteacher_s200.pkl,ckpt/teachers_noisy/dou_nteacher_s201.pkl,ckpt/teachers_noisy/dou_nteacher_s202.pkl,ckpt/teachers_noisy/dou_nteacher_s203.pkl,ckpt/teachers_noisy/dou_nteacher_s204.pkl,ckpt/teachers_noisy/dou_nteacher_s205.pkl,ckpt/teachers_noisy/dou_nteacher_s206.pkl,ckpt/teachers_noisy/dou_nteacher_s207.pkl"

# ============ PHASE A: complete teachers 204,205 (were killed) on known-idle GPU7 ============
for sd in 204 205; do
  if ! pgrep -f "dou_data_noisy.npz --seed $sd " >/dev/null; then
    CUDA_VISIBLE_DEVICES=7 nohup python3 dou_bc_train.py --data dou_data_noisy.npz \
        --seed $sd --steps 60000 --out ckpt/teachers_noisy/dou_nteacher_s${sd}.pkl \
        > logs/nteacher_s${sd}.log 2>&1 &
    log "(re)launched teacher seed=$sd on GPU=7 pid=$!"; sleep 20
  else
    log "teacher seed=$sd already running; skip"
  fi
done

# ============ PHASE B: wait for all 8 noisy teachers to finish ============
log "waiting for 8 noisy teachers (200-207) to finish"
until [ -f ckpt/teachers_noisy/dou_nteacher_s207.pkl ] && [ "$(pgrep -cf 'out ckpt/teachers_noisy/dou_nteacher_s')" = "0" ]; do sleep 20; done
log "all 8 noisy teachers finished"

# ============ PHASE C (PRIORITY): noisy H1 — 3 students (210-212) distilled from all 8 ============
for sd in 210 211 212; do
  launch_job logs/nstudent_s${sd}.log python3 dou_kd_train.py --data dou_data_noisy.npz \
      --teachers "$T_ALL" --seed $sd --steps 25000 --alpha 0.7 \
      --out ckpt/students_noisy/dou_nstudent_s${sd}.pkl
done
until [ -f ckpt/students_noisy/dou_nstudent_s212.pkl ] && [ "$(pgrep -cf 'out ckpt/students_noisy/dou_nstudent_s')" = "0" ]; do sleep 20; done
log "noisy H1 students finished; running H1 3-way gate"
launch_job logs/gate_noisy.log python3 run_gate3.py --regime noisy_eps0.3 \
  --single ckpt/students_noisy/dou_nstudent_s210.pkl \
  --teacher_ens ckpt/teachers_noisy/dou_nteacher_s200.pkl,ckpt/teachers_noisy/dou_nteacher_s201.pkl,ckpt/teachers_noisy/dou_nteacher_s202.pkl \
  --student_ens ckpt/students_noisy/dou_nstudent_s210.pkl,ckpt/students_noisy/dou_nstudent_s211.pkl,ckpt/students_noisy/dou_nstudent_s212.pkl \
  --seat 0 --nseeds 2000 --out results/noisy_gate.json
until [ -f results/noisy_gate.json ]; do sleep 10; done
log "NOISY GATE JSON WRITTEN -> results/noisy_gate.json"

# ============ PHASE D: teacher-count curve N in {1,2,4,8} ============
declare -A NTEA
NTEA[1]="ckpt/teachers_noisy/dou_nteacher_s200.pkl"
NTEA[2]="ckpt/teachers_noisy/dou_nteacher_s200.pkl,ckpt/teachers_noisy/dou_nteacher_s201.pkl"
NTEA[4]="ckpt/teachers_noisy/dou_nteacher_s200.pkl,ckpt/teachers_noisy/dou_nteacher_s201.pkl,ckpt/teachers_noisy/dou_nteacher_s202.pkl,ckpt/teachers_noisy/dou_nteacher_s203.pkl"
NTEA[8]="$T_ALL"
# base student seeds per N: 40*N .. +2   (distinct across N)
declare -A SEEDS
SEEDS[1]="4010 4011 4012"; SEEDS[2]="4020 4021 4022"; SEEDS[4]="4040 4041 4042"; SEEDS[8]="4080 4081 4082"
# launch ALL 12 curve students greedily across free GPUs
for N in 1 2 4 8; do
  for sd in ${SEEDS[$N]}; do
    launch_job logs/cstudent_N${N}_s${sd}.log python3 dou_kd_train.py --data dou_data_noisy.npz \
        --teachers "${NTEA[$N]}" --seed $sd --steps 25000 --alpha 0.7 \
        --out ckpt/students_curve/dou_cstudent_N${N}_s${sd}.pkl
  done
done
log "all 12 curve students launched; waiting for completion"
until [ "$(ls ckpt/students_curve/dou_cstudent_N*_s*.pkl 2>/dev/null | wc -l)" -ge 12 ] && [ "$(pgrep -cf 'out ckpt/students_curve/dou_cstudent_N')" = "0" ]; do sleep 20; done
log "all curve students finished; computing calibrated reference + per-N gates (3000 seeds)"

# calibrated reference = the H1 single student (fixed net, same seed set)
launch_job logs/curve_ref.log python3 gate_curve.py --tag ref \
  --pkls ckpt/students_noisy/dou_nstudent_s210.pkl --nseeds 3000 --out results/curve_ref.json
# per-N 3-student distill-ensemble gates
for N in 1 2 4 8; do
  ens=""; for sd in ${SEEDS[$N]}; do ens="$ens,ckpt/students_curve/dou_cstudent_N${N}_s${sd}.pkl"; done; ens=${ens#,}
  launch_job logs/curve_N${N}.log python3 gate_curve.py --tag N$N --pkls "$ens" --nseeds 3000 --out results/curve_N${N}.json
done
until [ -f results/curve_ref.json ] && [ -f results/curve_N1.json ] && [ -f results/curve_N2.json ] && [ -f results/curve_N4.json ] && [ -f results/curve_N8.json ] && [ "$(pgrep -cf 'gate_curve.py')" = "0" ]; do sleep 15; done

python3 - <<'PY'
import json
out={"regime":"noisy_eps0.3","metric":"mean_payoff seat0 vs 2 rule agents","nseeds":3000}
r=json.load(open("results/curve_ref.json"))
out["calibrated_reference"]={"pkl":"dou_nstudent_s210 (single)","mean_payoff":r["mean_payoff"],"ci95":r["ci95"]}
out["curve"]={}
for N in (1,2,4,8):
    d=json.load(open(f"results/curve_N{N}.json"))
    out["curve"][str(N)]={"mean_payoff":d["mean_payoff"],"ci95":d["ci95"],"K_students":d["K"]}
json.dump(out,open("results/noisy_curve.json","w"),indent=2)
print(json.dumps(out,indent=2))
PY
log "NOISY CURVE JSON WRITTEN -> results/noisy_curve.json"
log "NOISY MASTER COMPLETE"
