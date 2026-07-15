#!/bin/bash
# NOISE-LEVEL (epsilon) SWEEP. CPU data-gen up front, then per-eps 6 teachers -> 6 students ->
# gate, back-to-back for eps in {0.1,0.2,0.5}. Coexists with repl fleet + curve via a memory-based
# picker (free GPU = mem<2000MB AND <3 compute procs). Never touches other jobs' pkls.
cd /root/crossgame/doudizhu || exit 1
mkdir -p ckpt/sweep logs results
log(){ echo "[sweep $(date +%F_%T)] $*"; }

launch_job(){  # $1=logfile ; rest=command. Blocks until a free GPU (mem<2000MB & <3 procs).
  local lf="$1"; shift; local g="" used napps cand
  while [ -z "$g" ]; do
    for cand in 0 1 2 3 4 5 6 7; do
      used=$(nvidia-smi -i $cand --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
      [ -z "$used" ] && continue
      napps=$(nvidia-smi -i $cand --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
      if [ "$used" -lt 2000 ] && [ "$napps" -lt 3 ]; then g=$cand; break; fi
    done
    [ -z "$g" ] && sleep 15
  done
  CUDA_VISIBLE_DEVICES=$g nohup "$@" > "$lf" 2>&1 &
  log "launched on GPU=$g -> $lf"
  sleep 30
}

# ---- Phase 0: data-gen for eps 0.1,0.2,0.5 (CPU, ahead of GPU training) ----
gen(){  # $1=eps  $2=tag  $3=genseed
  if [ ! -f dou_data_eps$2.npz ]; then
    python3 gen_dou_data.py --games 11000 --workers 64 --seed $3 --epsilon $1 --out dou_data_eps$2.npz >> logs/sweep_gen.log 2>&1
  fi
  log "data eps$1 ready -> dou_data_eps$2.npz"
}
gen 0.1 01 8100
gen 0.2 02 8200
gen 0.5 05 8500

EPS=(0.1 0.2 0.5); TAG=(01 02 05); TBASE=(7100 7200 7500); SBASE=(7110 7210 7510)
for k in 0 1 2; do
  e=${EPS[$k]}; t=${TAG[$k]}; tb=${TBASE[$k]}; sb=${SBASE[$k]}
  d=ckpt/sweep/e$t; mkdir -p $d
  # ----- 6 teachers -----
  log "=== eps=$e : launching 6 teachers ==="
  for j in 1 2 3 4 5 6; do
    sd=$((tb+j))
    launch_job logs/sweep_e${t}_tea_s${sd}.log python3 dou_bc_train.py --data dou_data_eps$t.npz \
        --seed $sd --steps 60000 --out $d/tea_s${sd}.pkl
  done
  until [ "$(ls $d/tea_s*.pkl 2>/dev/null | wc -l)" -ge 6 ] && [ "$(pgrep -cf "out $d/tea_s")" = "0" ]; do sleep 20; done
  log "eps=$e teachers done"
  TEA="$d/tea_s$((tb+1)).pkl,$d/tea_s$((tb+2)).pkl,$d/tea_s$((tb+3)).pkl,$d/tea_s$((tb+4)).pkl,$d/tea_s$((tb+5)).pkl,$d/tea_s$((tb+6)).pkl"
  # ----- 6 students (distilled from all 6 teachers) -----
  log "=== eps=$e : launching 6 students ==="
  for j in 1 2 3 4 5 6; do
    sd=$((sb+j))
    launch_job logs/sweep_e${t}_stu_s${sd}.log python3 dou_kd_train.py --data dou_data_eps$t.npz \
        --teachers "$TEA" --seed $sd --steps 25000 --alpha 0.7 --out $d/stu_s${sd}.pkl
  done
  until [ "$(ls $d/stu_s*.pkl 2>/dev/null | wc -l)" -ge 6 ] && [ "$(pgrep -cf "out $d/stu_s")" = "0" ]; do sleep 20; done
  log "eps=$e students done; gating (3000 seeds)"
  # ----- gate: single / seed-ens(3 teachers) / distill-ens A & B -----
  launch_job logs/sweep_e${t}_gate.log python3 sweep_gate.py --tag $e \
    --single $d/stu_s$((sb+1)).pkl \
    --seed_teachers $d/tea_s$((tb+1)).pkl,$d/tea_s$((tb+2)).pkl,$d/tea_s$((tb+3)).pkl \
    --distillA $d/stu_s$((sb+1)).pkl,$d/stu_s$((sb+2)).pkl,$d/stu_s$((sb+3)).pkl \
    --distillB $d/stu_s$((sb+4)).pkl,$d/stu_s$((sb+5)).pkl,$d/stu_s$((sb+6)).pkl \
    --nseeds 3000 --seed0 10000 --out results/sweep_e$t.json
  until [ -f results/sweep_e$t.json ]; do sleep 10; done
  log "eps=$e gate done -> results/sweep_e$t.json"
done

# ---- merge into noise_sweep.json (eps 0 and 0.3 pulled from existing gates) ----
python3 - <<'PY'
import json
def g3(f):
    d=json.load(open(f))
    return {"single":d["single_mean_payoff"],"seed_ens":d["teacher_ens3_mean_payoff"],
            "distill_ens":d["student_ens3_mean_payoff"],
            "gap":round(d["student_ens3_mean_payoff"]-d["teacher_ens3_mean_payoff"],5)}
by={}
by["0"]=g3("results/det_gate.json")
by["0.3"]=g3("results/noisy_gate.json")
for t,e in (("01","0.1"),("02","0.2"),("05","0.5")):
    d=json.load(open(f"results/sweep_e{t}.json"))
    by[e]={"single":d["single"],"seed_ens":d["seed_ens"],"distill_ens":d["distill_ens"],
           "gap":d["gap"],"gap_ci95":d.get("gap_ci95"),"distill_ensB":d.get("distill_ensB"),
           "calibration_delta":d.get("calibration_delta")}
res={"experiment":"noise_level_epsilon_sweep",
     "metric":"mean_payoff seat0 vs 2 rule agents",
     "note":"eps0<-det_gate(2000 seeds); eps0.3<-noisy_gate(2000 seeds); eps0.1/0.2/0.5 fresh(3000 seeds, per-game-seeded)",
     "hypothesis":"distill-then-ensemble gap over seed-ensemble grows with imitation-target noise eps",
     "by_eps":{k:by[k] for k in ["0","0.1","0.2","0.3","0.5"]}}
json.dump(res,open("results/noise_sweep.json","w"),indent=2)
print(json.dumps(res,indent=2))
PY
log "NOISE SWEEP JSON WRITTEN -> results/noise_sweep.json"
log "NOISE SWEEP COMPLETE"
