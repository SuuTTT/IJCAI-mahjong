#!/bin/bash
# JOINT-DEFENSE runner. Chains behind the M replication (never touches its jobs):
#   1. wait for /root/se_mahjong/results/M_REPL_DONE
#   2. KD pool: 4 lambdas x 3 seeds = 12 trainers (60k steps, 8 GPUs, per=1)
#   3. deal-in pool (GPU, 24 jobs) + placement gate loop (CPU, 32 blocks) CONCURRENTLY
#   4. JD_agg.py -> results/JOINT_DEFENSE.json ; touch results/JD_DONE
cd /root/caiest_repro
export PYTHONUNBUFFERED=1
LOG=/root/caiest_repro/logs_jd.log
mkdir -p ckpt/jd results/jd_gate results/jd_dealin logs
echo "[$(date +%H:%M:%S)] JD runner start" >> $LOG

# 1. wait for replication to finish (rechecks every 2 min)
while [ ! -f /root/se_mahjong/results/M_REPL_DONE ]; do sleep 120; done
echo "[$(date +%H:%M:%S)] replication done -> KD pool" >> $LOG

# 2. KD training pool
python3 - <<'PY'
lams=["0","0.1","0.3","1.0"]; seeds=[0,1,2]
open('/root/caiest_repro/jd_train_jobs.txt','w').write('\n'.join(
  f'cd /root/caiest_repro && python3 e13_kd_danger.py --lam_danger {l} --seed {s} --steps 60000 '
  f'--out ckpt/jd/jd_lam{l}_s{s}.pkl' for l in lams for s in seeds)+'\n')
PY
python3 /root/se_mahjong/gpu_pool.py /root/caiest_repro/jd_train_jobs.txt --gpus 0,1,2,3,4,5,6,7 --per 1 >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] KD pool done" >> $LOG

# sanity: all 12 fused students exist
NPKL=$(ls ckpt/jd/jd_lam*_s*.pkl 2>/dev/null | grep -v bn | wc -l)
echo "students present: $NPKL/12" >> $LOG

# 3a. placement gate (CPU) in background: 4 lambdas x 8 blocks x 500 seeds
(
for L in 0 0.1 0.3 1.0; do
  CAND="ckpt/jd/jd_lam${L}_s0.pkl,ckpt/jd/jd_lam${L}_s1.pkl,ckpt/jd/jd_lam${L}_s2.pkl"
  for B in 0 1 2 3 4 5 6 7; do
    S0=$((300000 + B * 500))
    python3 e12_ens_gate.py --cand $CAND --ref ckpt/aug/aug_128x40_s0.pkl --seeds 500 \
      --workers 96 --seed0 $S0 --out results/jd_gate/lam${L}_b${B}.json >> $LOG 2>&1
  done
done
echo "[$(date +%H:%M:%S)] gate blocks done" >> $LOG
) &
GATEPID=$!

# 3b. deal-in pool (GPU): 4 lambdas x 6 eval seeds x 250 games
python3 - <<'PY'
lams=["0","0.1","0.3","1.0"]
jobs=[]
for l in lams:
    cand=f'ckpt/jd/jd_lam{l}_s0.pkl,ckpt/jd/jd_lam{l}_s1.pkl,ckpt/jd/jd_lam{l}_s2.pkl'
    for e in range(6):
        jobs.append(f'cd /root/caiest_repro && python3 jd_dealin_eval.py --cand {cand} '
                    f'--evalseed {e} --ngames 250 --out results/jd_dealin/lam{l}_e{e}.json')
open('/root/caiest_repro/jd_dealin_jobs.txt','w').write('\n'.join(jobs)+'\n')
PY
python3 /root/se_mahjong/gpu_pool.py /root/caiest_repro/jd_dealin_jobs.txt --gpus 0,1,2,3,4,5,6,7 --per 2 >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] deal-in pool done" >> $LOG
wait $GATEPID

# 4. aggregate
python3 JD_agg.py >> $LOG 2>&1
cp results/JOINT_DEFENSE.json /root/se_mahjong/results/JOINT_DEFENSE.json 2>/dev/null
touch results/JD_DONE /root/se_mahjong/results/JD_DONE
echo "[$(date +%H:%M:%S)] JD ALL DONE" >> $LOG
