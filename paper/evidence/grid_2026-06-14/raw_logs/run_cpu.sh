#!/bin/bash
set -u; cd /root/mahjong
SEED0="${SEED0:-600000}"; NSEED="${NSEED:-5}"; N="${N:-144}"; JOBS="${JOBS:-4}"
BC='CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 BOTZONE_JSON=0 CAIEST_PIMC=0 python3 __main__.py'
export MAHJONG_JUDGE=/workspace/Chinese-Standard-Mahjong/judge/judge BENCH_TIMEOUT=40 PYTHONPATH=/root/mahjong
mkdir -p res
rm -rf base base2; cp -r botA base; cp -r botA base2; mkdir -p base/data base2/data
ln -sf /root/mahjong/data/cnn_lad_chunjiandu.pkl base/data/cnn.pkl; ln -sf /root/mahjong/data/cnn_lad_chunjiandu.pkl base2/data/cnn.pkl
sem(){ while [ "$(jobs -rp|wc -l)" -ge "$JOBS" ]; do sleep 5; done; }
bench(){ WALL_SEED_BASE=$4 timeout 6000 python3 eval/bench_vs_bot.py "cd /root/mahjong/$1 && $BC" "cd /root/mahjong/$3 && $BC" "$N" "$2" opp > "res/$2.log" 2>&1; }
# noise floor
for i in $(seq 0 $((NSEED-1))); do sd=$((SEED0+i*1000)); [ -f res/nf_$sd.log ] && grep -q requested, res/nf_$sd.log && continue; sem; bench base nf_$sd base2 $sd & done
# gauntlet candidates as they appear (loop until GPU loop done and all benched)
GSEED=$((SEED0+50000))
while true; do
  for pkl in cand_*.pkl; do [ -e "$pkl" ] || continue; tag="${pkl%.pkl}"; [ -f "res/g5_${tag#cand_}.log" ] && continue
    d="bd_${tag#cand_}"; rm -rf "$d"; cp -r botA "$d"; mkdir -p "$d/data"; ln -sf "/root/mahjong/$pkl" "$d/data/cnn.pkl"
    sem; bench "$d" "g5_${tag#cand_}" base "$GSEED" & done
  grep -q GPU_LOOP_DONE res/GPU.log 2>/dev/null && [ -z "$(for p in cand_*.pkl; do [ -e "$p" ] && t="${p%.pkl}"; [ -f "res/g5_${t#cand_}.log" ]||echo x; done)" ] && break
  sleep 60
done
wait; echo "CPU_LOOP_DONE $(date -u)" >> res/CPU.log
