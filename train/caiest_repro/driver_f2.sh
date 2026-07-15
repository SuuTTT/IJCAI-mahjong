#!/bin/bash
# driver_f2.sh — Final2 campaign orchestrator (runs under setsid, fully autonomous).
# Prereq (done synchronously before launch): final2_cai_corpus.npz built+verified,
# DANGER_STD.json written, score-gate calibration passed.
#
# GPU pool (8x A4000, 17 jobs): value_frozen | armd x3 | armb x3 | armc x3 |
#   jdv2 {0.3,1.0,3.0} x {s0,s1} | value_e2e
# CPU lane A: score-gate validation (12 blocks) -> armb gates (12) -> jdv2 gates
# CPU lane B: armd gates (12) -> armc gates (12) -> lam0-2stu gate (8)
# After GPU pool: jdv2+lam0 deal-in evals (GPU) -> aggregators -> DONE flags.
cd /root/caiest_repro
export PYTHONUNBUFFERED=1
LOG=/root/caiest_repro/logs/f2_driver.log
mkdir -p ckpt/f2 ckpt/jdv2 results/f2_gate results/jdv2_gate results/jdv2_dealin \
         results/score_gate_val logs
say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> $LOG; }
say "F2 driver start"

KDENS3=ckpt/kd/kd_128x40_s0.pkl,ckpt/kd/kd_128x40_s1.pkl,ckpt/kd/kd_128x40_s2.pkl
AUG0=ckpt/aug/aug_128x40_s0.pkl
CORPUS=/root/final2_harvest/final2_cai_corpus.npz

# ---------------- trainer jobs ----------------
cat > /root/caiest_repro/f2_train_jobs.txt <<EOF
cd /root/caiest_repro && python3 f2_value_head.py --mode frozen --out results/value_frozen.json > logs/f2_value_frozen.log 2>&1
cd /root/caiest_repro && python3 e13_kd_corpus.py --pure --seed 0 --steps 60000 --out ckpt/f2/armd_s0.pkl > logs/f2_armd_s0.log 2>&1
cd /root/caiest_repro && python3 e13_kd_corpus.py --pure --seed 1 --steps 60000 --out ckpt/f2/armd_s1.pkl > logs/f2_armd_s1.log 2>&1
cd /root/caiest_repro && python3 e13_kd_corpus.py --pure --seed 2 --steps 60000 --out ckpt/f2/armd_s2.pkl > logs/f2_armd_s2.log 2>&1
cd /root/caiest_repro && python3 e13_kd_corpus.py --beta 0.3 --seed 0 --steps 60000 --out ckpt/f2/armb_s0.pkl > logs/f2_armb_s0.log 2>&1
cd /root/caiest_repro && python3 e13_kd_corpus.py --beta 0.3 --seed 1 --steps 60000 --out ckpt/f2/armb_s1.pkl > logs/f2_armb_s1.log 2>&1
cd /root/caiest_repro && python3 e13_kd_corpus.py --beta 0.3 --seed 2 --steps 60000 --out ckpt/f2/armb_s2.pkl > logs/f2_armb_s2.log 2>&1
cd /root/caiest_repro && python3 e13_kd_corpus.py --beta 0.3 --bots 0,1 --seed 0 --steps 60000 --out ckpt/f2/armc_s0.pkl > logs/f2_armc_s0.log 2>&1
cd /root/caiest_repro && python3 e13_kd_corpus.py --beta 0.3 --bots 0,1 --seed 1 --steps 60000 --out ckpt/f2/armc_s1.pkl > logs/f2_armc_s1.log 2>&1
cd /root/caiest_repro && python3 e13_kd_corpus.py --beta 0.3 --bots 0,1 --seed 2 --steps 60000 --out ckpt/f2/armc_s2.pkl > logs/f2_armc_s2.log 2>&1
cd /root/caiest_repro && python3 e13_kd_danger_v2.py --lam_danger 0.3 --seed 0 --steps 60000 --out ckpt/jdv2/jdv2_lam0.3_s0.pkl > logs/jdv2_l0.3_s0.log 2>&1
cd /root/caiest_repro && python3 e13_kd_danger_v2.py --lam_danger 0.3 --seed 1 --steps 60000 --out ckpt/jdv2/jdv2_lam0.3_s1.pkl > logs/jdv2_l0.3_s1.log 2>&1
cd /root/caiest_repro && python3 e13_kd_danger_v2.py --lam_danger 1.0 --seed 0 --steps 60000 --out ckpt/jdv2/jdv2_lam1.0_s0.pkl > logs/jdv2_l1.0_s0.log 2>&1
cd /root/caiest_repro && python3 e13_kd_danger_v2.py --lam_danger 1.0 --seed 1 --steps 60000 --out ckpt/jdv2/jdv2_lam1.0_s1.pkl > logs/jdv2_l1.0_s1.log 2>&1
cd /root/caiest_repro && python3 e13_kd_danger_v2.py --lam_danger 3.0 --seed 0 --steps 60000 --out ckpt/jdv2/jdv2_lam3.0_s0.pkl > logs/jdv2_l3.0_s0.log 2>&1
cd /root/caiest_repro && python3 e13_kd_danger_v2.py --lam_danger 3.0 --seed 1 --steps 60000 --out ckpt/jdv2/jdv2_lam3.0_s1.pkl > logs/jdv2_l3.0_s1.log 2>&1
cd /root/caiest_repro && python3 f2_value_head.py --mode e2e --out results/value_e2e.json > logs/f2_value_e2e.log 2>&1
EOF
python3 /root/se_mahjong/gpu_pool.py /root/caiest_repro/f2_train_jobs.txt \
  --gpus 0,1,2,3,4,5,6,7 --per 1 >> $LOG 2>&1 &
POOLPID=$!
say "GPU pool launched pid=$POOLPID (17 jobs)"

wait_files() {  # wait_files <timeout_s> f1 f2 ...
  local T=$1; shift
  local t=0
  while :; do
    local ok=1
    for f in "$@"; do [ -f "$f" ] || ok=0; done
    [ $ok -eq 1 ] && return 0
    [ $t -ge $T ] && return 1
    sleep 120; t=$((t+120))
  done
}

dual_gate_blocks() {  # dual_gate_blocks <cand> <ref> <nblocks> <seed0base> <outprefix> <workers>
  local CAND=$1 REF=$2 NB=$3 S0B=$4 OUT=$5 W=$6
  for ((i=0; i<NB; i++)); do
    [ -f ${OUT}_b${i}.json ] || \
      python3 e12_score_gate.py --cand $CAND --ref $REF --seeds 500 --workers $W \
        --seed0 $((S0B + i*500)) --out ${OUT}_b${i}.json >> $LOG 2>&1
  done
}

# ---------------- CPU lane A ----------------
(
say "laneA: score-gate validation (kdens3 vs aug_s0, 12 blocks)"
dual_gate_blocks $KDENS3 $AUG0 12 500000 results/score_gate_val/b 64
python3 f2_aggregate.py score >> $LOG 2>&1
say "laneA: SCORE_GATE.json written"

B=ckpt/f2/armb_s0.pkl,ckpt/f2/armb_s1.pkl,ckpt/f2/armb_s2.pkl
if wait_files 86400 ckpt/f2/armb_s0.pkl ckpt/f2/armb_s1.pkl ckpt/f2/armb_s2.pkl; then
  say "laneA: armb gates start"
  dual_gate_blocks $B $KDENS3 12 500000 results/f2_gate/armb_b 64
  say "laneA: armb gates done"
else say "laneA: TIMEOUT waiting armb students"; fi

J0=ckpt/jdv2
if wait_files 130000 $J0/jdv2_lam0.3_s0.pkl $J0/jdv2_lam0.3_s1.pkl $J0/jdv2_lam1.0_s0.pkl \
               $J0/jdv2_lam1.0_s1.pkl $J0/jdv2_lam3.0_s0.pkl $J0/jdv2_lam3.0_s1.pkl; then
  say "laneA: jdv2 gates start"
  for L in 0.3 1.0 3.0; do
    dual_gate_blocks $J0/jdv2_lam${L}_s0.pkl,$J0/jdv2_lam${L}_s1.pkl $AUG0 8 300000 \
      results/jdv2_gate/lam${L}_b 64
  done
  say "laneA: jdv2 gates done"
else say "laneA: TIMEOUT waiting jdv2 students"; fi
) &
LANEA=$!

# ---------------- CPU lane B ----------------
(
if wait_files 86400 ckpt/f2/armd_s0.pkl ckpt/f2/armd_s1.pkl ckpt/f2/armd_s2.pkl; then
  say "laneB: armd gates start"
  D=ckpt/f2/armd_s0.pkl,ckpt/f2/armd_s1.pkl,ckpt/f2/armd_s2.pkl
  dual_gate_blocks $D $KDENS3 12 500000 results/f2_gate/armd_b 56
  say "laneB: armd gates done"
else say "laneB: TIMEOUT waiting armd students"; fi

if wait_files 86400 ckpt/f2/armc_s0.pkl ckpt/f2/armc_s1.pkl ckpt/f2/armc_s2.pkl; then
  say "laneB: armc gates start"
  C=ckpt/f2/armc_s0.pkl,ckpt/f2/armc_s1.pkl,ckpt/f2/armc_s2.pkl
  dual_gate_blocks $C $KDENS3 12 500000 results/f2_gate/armc_b 56
  say "laneB: armc gates done"
else say "laneB: TIMEOUT waiting armc students"; fi

say "laneB: lam0 2-student control gate start"
dual_gate_blocks ckpt/jd/jd_lam0_s0.pkl,ckpt/jd/jd_lam0_s1.pkl $AUG0 8 300000 \
  results/jdv2_gate/lam0_b 56
say "laneB: lam0 control gate done"
) &
LANEB=$!

wait $POOLPID
say "GPU pool done -> value agg + deal-in evals"
python3 f2_aggregate.py value >> $LOG 2>&1

# ---------------- deal-in evals (GPU) ----------------
python3 - <<'PY'
jobs = []
J0 = "ckpt/jdv2"
for lam in ["0.3", "1.0", "3.0"]:
    cand = f"{J0}/jdv2_lam{lam}_s0.pkl,{J0}/jdv2_lam{lam}_s1.pkl"
    for e in range(6):
        jobs.append(f"cd /root/caiest_repro && python3 jd_dealin_eval.py --cand {cand} "
                    f"--evalseed {e} --ngames 250 --out results/jdv2_dealin/lam{lam}_e{e}.json")
cand0 = "ckpt/jd/jd_lam0_s0.pkl,ckpt/jd/jd_lam0_s1.pkl"
for e in range(6):
    jobs.append(f"cd /root/caiest_repro && python3 jd_dealin_eval.py --cand {cand0} "
                f"--evalseed {e} --ngames 250 --out results/jdv2_dealin/lam0_e{e}.json")
open("/root/caiest_repro/jdv2_dealin_jobs.txt", "w").write("\n".join(jobs) + "\n")
PY
python3 /root/se_mahjong/gpu_pool.py /root/caiest_repro/jdv2_dealin_jobs.txt \
  --gpus 0,1,2,3,4,5,6,7 --per 2 >> $LOG 2>&1
say "deal-in evals done"

wait $LANEA $LANEB
say "CPU lanes done -> final aggregation"
python3 f2_aggregate.py exp1 >> $LOG 2>&1
python3 f2_aggregate.py jdv2 >> $LOG 2>&1
python3 f2_aggregate.py score >> $LOG 2>&1
python3 f2_aggregate.py value >> $LOG 2>&1
touch results/F2_CAMPAIGN_DONE
say "F2 CAMPAIGN ALL DONE"
