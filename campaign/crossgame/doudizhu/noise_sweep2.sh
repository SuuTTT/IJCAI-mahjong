#!/bin/bash
# PIPELINED noise-level sweep: the 3 eps tracks run CONCURRENTLY so GPUs never idle at a
# per-eps barrier. eps=0.1 teachers are already running (launched by the previous orchestrator),
# so that track only waits + trains students. Shared LOCKED memory-picker (free = mem<2000MB &
# <3 procs) prevents oversubscription races between tracks. Never touches other jobs' pkls.
cd /root/crossgame/doudizhu || exit 1
mkdir -p ckpt/sweep logs results
log(){ echo "[sweep2 $(date +%F_%T)] $*"; }
LOCK=/tmp/sweep2_gpu.lock

launch_job(){  # $1=logfile ; rest=command. Locked pick of a free GPU (mem<2000 & <3 procs).
  local lf="$1"; shift; local g="" used napps cand
  while ! mkdir "$LOCK" 2>/dev/null; do sleep 2; done   # serialize selection across tracks
  while [ -z "$g" ]; do
    for cand in 0 1 2 3 4 5 6 7; do
      used=$(nvidia-smi -i $cand --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
      [ -z "$used" ] && continue
      napps=$(nvidia-smi -i $cand --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
      if [ "$used" -lt 2000 ] && [ "$napps" -lt 3 ]; then g=$cand; break; fi
    done
    [ -z "$g" ] && sleep 12
  done
  CUDA_VISIBLE_DEVICES=$g nohup "$@" > "$lf" 2>&1 &
  log "launched on GPU=$g -> $lf"
  sleep 22            # let CUDA ctx + data load register before releasing the lock
  rmdir "$LOCK"
}

run_eps(){  # $1=tag $2=eps $3=tbase $4=sbase $5=launch_teachers(0/1)
  local t=$1 e=$2 tb=$3 sb=$4 LT=$5 d=ckpt/sweep/e$1
  mkdir -p $d
  if [ "$LT" = "1" ]; then
    log "eps=$e launching 6 teachers"
    for j in 1 2 3 4 5 6; do
      launch_job logs/sweep_e${t}_tea_s$((tb+j)).log python3 dou_bc_train.py --data dou_data_eps$t.npz \
          --seed $((tb+j)) --steps 60000 --out $d/tea_s$((tb+j)).pkl
    done
  else
    log "eps=$e teachers already running from prior orchestrator; waiting"
  fi
  until [ "$(ls $d/tea_s*.pkl 2>/dev/null | wc -l)" -ge 6 ] && [ "$(pgrep -cf "out $d/tea_s")" = "0" ]; do sleep 20; done
  log "eps=$e teachers done -> students"
  local TEA="$d/tea_s$((tb+1)).pkl,$d/tea_s$((tb+2)).pkl,$d/tea_s$((tb+3)).pkl,$d/tea_s$((tb+4)).pkl,$d/tea_s$((tb+5)).pkl,$d/tea_s$((tb+6)).pkl"
  for j in 1 2 3 4 5 6; do
    launch_job logs/sweep_e${t}_stu_s$((sb+j)).log python3 dou_kd_train.py --data dou_data_eps$t.npz \
        --teachers "$TEA" --seed $((sb+j)) --steps 25000 --alpha 0.7 --out $d/stu_s$((sb+j)).pkl
  done
  until [ "$(ls $d/stu_s*.pkl 2>/dev/null | wc -l)" -ge 6 ] && [ "$(pgrep -cf "out $d/stu_s")" = "0" ]; do sleep 20; done
  log "eps=$e students done -> gate"
  launch_job logs/sweep_e${t}_gate.log python3 sweep_gate.py --tag $e \
    --single $d/stu_s$((sb+1)).pkl \
    --seed_teachers $d/tea_s$((tb+1)).pkl,$d/tea_s$((tb+2)).pkl,$d/tea_s$((tb+3)).pkl \
    --distillA $d/stu_s$((sb+1)).pkl,$d/stu_s$((sb+2)).pkl,$d/stu_s$((sb+3)).pkl \
    --distillB $d/stu_s$((sb+4)).pkl,$d/stu_s$((sb+5)).pkl,$d/stu_s$((sb+6)).pkl \
    --nseeds 3000 --seed0 10000 --out results/sweep_e$t.json
  until [ -f results/sweep_e$t.json ]; do sleep 10; done
  log "eps=$e GATE DONE -> results/sweep_e$t.json"
}

# 3 concurrent tracks: eps0.1 teachers already running (LT=0); eps0.2,0.5 launch teachers (LT=1)
run_eps 01 0.1 7100 7110 0 &
run_eps 02 0.2 7200 7210 1 &
run_eps 05 0.5 7500 7510 1 &
wait
log "all 3 eps tracks complete; merging"

python3 - <<'PY'
import json
def g3(f):
    d=json.load(open(f)); return {"single":d["single_mean_payoff"],"seed_ens":d["teacher_ens3_mean_payoff"],
        "distill_ens":d["student_ens3_mean_payoff"],"gap":round(d["student_ens3_mean_payoff"]-d["teacher_ens3_mean_payoff"],5)}
by={"0":g3("results/det_gate.json"),"0.3":g3("results/noisy_gate.json")}
for t,e in (("01","0.1"),("02","0.2"),("05","0.5")):
    d=json.load(open(f"results/sweep_e{t}.json"))
    by[e]={"single":d["single"],"seed_ens":d["seed_ens"],"distill_ens":d["distill_ens"],
           "gap":d["gap"],"gap_ci95":d.get("gap_ci95"),"distill_ensB":d.get("distill_ensB"),
           "calibration_delta":d.get("calibration_delta")}
res={"experiment":"noise_level_epsilon_sweep","metric":"mean_payoff seat0 vs 2 rule agents",
     "note":"eps0<-det_gate(2000s); eps0.3<-noisy_gate(2000s); eps0.1/0.2/0.5 fresh(3000s per-game-seeded)",
     "hypothesis":"distill-then-ensemble gap over seed-ensemble grows with imitation-target noise eps",
     "by_eps":{k:by[k] for k in ["0","0.1","0.2","0.3","0.5"]}}
json.dump(res,open("results/noise_sweep.json","w"),indent=2); print(json.dumps(res,indent=2))
PY
log "NOISE SWEEP JSON WRITTEN -> results/noise_sweep.json"
log "NOISE SWEEP COMPLETE"
