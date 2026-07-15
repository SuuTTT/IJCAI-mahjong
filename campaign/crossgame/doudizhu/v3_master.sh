#!/bin/bash
# Audit M1 fix: protocol-matched eps=0 and eps=0.3 legs, EXACTLY the noise_sweep protocol:
# gen 11000 games at eps -> 6 BC teachers (60k steps) -> 6 KD students (25k steps, alpha .7,
# distilled from all 6 teachers) -> trio gate (seed_trio1/2 + distillA/B) on seeds 10000-12999.
# GPU picker: only GPUs with >8GB free and <3 compute procs (coexist with se_mahjong grid;
# NEVER kill anything). Then merge fully-uniform noise_sweep_v3.json.
cd /root/crossgame/doudizhu || exit 1
mkdir -p ckpt/sweep/e00s ckpt/sweep/e03s logs results
log(){ echo "[v3 $(date +%F_%T)] $*"; }
LOCK=/tmp/v3_gpu.lock

launch_job(){  # $1=logfile ; rest=command. Locked pick of a GPU with >8GB free & <3 procs.
  local lf="$1"; shift; local g="" free napps cand
  while ! mkdir "$LOCK" 2>/dev/null; do sleep 3; done
  while [ -z "$g" ]; do
    for cand in 0 1 2 3 4 5 6 7; do
      free=$(nvidia-smi -i $cand --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
      [ -z "$free" ] && continue
      napps=$(nvidia-smi -i $cand --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
      if [ "$free" -gt 8000 ] && [ "$napps" -lt 3 ]; then g=$cand; break; fi
    done
    [ -z "$g" ] && sleep 15
  done
  CUDA_VISIBLE_DEVICES=$g nohup "$@" > "$lf" 2>&1 &
  log "launched on GPU=$g -> $lf"
  sleep 25
  rmdir "$LOCK"
}

# ---- data (CPU, sequential): same sample size as sweep eps (11000 games ~ 516k) ----
if [ ! -f dou_data_eps00.npz ]; then
  log "generating dou_data_eps00.npz (eps=0.0, 11000 games)"
  python3 gen_dou_data.py --games 11000 --workers 64 --seed 8000 --epsilon 0.0 --out dou_data_eps00.npz >> logs/v3_gen.log 2>&1
fi
if [ ! -f dou_data_eps03.npz ]; then
  log "generating dou_data_eps03.npz (eps=0.3, 11000 games)"
  python3 gen_dou_data.py --games 11000 --workers 64 --seed 8300 --epsilon 0.3 --out dou_data_eps03.npz >> logs/v3_gen.log 2>&1
fi
log "data ready: $(python3 -c "import numpy as np; print({f: np.load(f)['act'].shape[0] for f in ['dou_data_eps00.npz','dou_data_eps03.npz']})")"

run_eps(){  # $1=dirtag $2=eps $3=datafile $4=tbase $5=sbase
  local t=$1 e=$2 df=$3 tb=$4 sb=$5 d=ckpt/sweep/e$1
  log "=== eps=$e ($t): 6 teachers ==="
  for j in 1 2 3 4 5 6; do
    [ -f $d/tea_s$((tb+j)).pkl ] || launch_job logs/v3_e${t}_tea_s$((tb+j)).log \
      python3 dou_bc_train.py --data $df --seed $((tb+j)) --steps 60000 --out $d/tea_s$((tb+j)).pkl
  done
  until [ "$(ls $d/tea_s*.pkl 2>/dev/null | wc -l)" -ge 6 ] && [ "$(pgrep -cf "out $d/tea_s")" = "0" ]; do sleep 30; done
  log "eps=$e teachers done"
  local TEA="$d/tea_s$((tb+1)).pkl,$d/tea_s$((tb+2)).pkl,$d/tea_s$((tb+3)).pkl,$d/tea_s$((tb+4)).pkl,$d/tea_s$((tb+5)).pkl,$d/tea_s$((tb+6)).pkl"
  log "=== eps=$e ($t): 6 students ==="
  for j in 1 2 3 4 5 6; do
    [ -f $d/stu_s$((sb+j)).pkl ] || launch_job logs/v3_e${t}_stu_s$((sb+j)).log \
      python3 dou_kd_train.py --data $df --teachers "$TEA" --seed $((sb+j)) --steps 25000 --alpha 0.7 --out $d/stu_s$((sb+j)).pkl
  done
  until [ "$(ls $d/stu_s*.pkl 2>/dev/null | wc -l)" -ge 6 ] && [ "$(pgrep -cf "out $d/stu_s")" = "0" ]; do sleep 30; done
  log "eps=$e students done -> trio gate"
  launch_job logs/v3_e${t}_gate.log python3 regate_trios.py --tag $e \
    --ens seed_trio1=$d/tea_s$((tb+1)).pkl,$d/tea_s$((tb+2)).pkl,$d/tea_s$((tb+3)).pkl \
    --ens seed_trio2=$d/tea_s$((tb+4)).pkl,$d/tea_s$((tb+5)).pkl,$d/tea_s$((tb+6)).pkl \
    --ens distillA=$d/stu_s$((sb+1)).pkl,$d/stu_s$((sb+2)).pkl,$d/stu_s$((sb+3)).pkl \
    --ens distillB=$d/stu_s$((sb+4)).pkl,$d/stu_s$((sb+5)).pkl,$d/stu_s$((sb+6)).pkl \
    --nseeds 3000 --seed0 10000 --out results/regate_${t}_v3.json --npz results/regate_${t}_v3.npz
  until grep -q "DONE eps=" logs/v3_e${t}_gate.log 2>/dev/null; do sleep 30; done
  log "eps=$e ($t) V3 GATE DONE -> results/regate_${t}_v3.json"
}

run_eps 00s 0   dou_data_eps00.npz 7000 7010 &
run_eps 03s 0.3 dou_data_eps03.npz 7300 7310 &
wait
log "both v3 legs gated; merging noise_sweep_v3.json"
python3 merge_noise_sweep_v3.py > logs/v3_merge.log 2>&1 && log "V3 MERGE OK" || log "V3 MERGE FAILED"
log "V3 COMPLETE"
