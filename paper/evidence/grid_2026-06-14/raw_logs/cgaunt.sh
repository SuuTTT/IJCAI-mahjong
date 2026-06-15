#!/bin/bash
cd /root/mahjong
BC='CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 BOTZONE_JSON=0 CAIEST_PIMC=0 python3 __main__.py'
export MAHJONG_JUDGE=/workspace/Chinese-Standard-Mahjong/judge/judge BENCH_TIMEOUT=40 PYTHONPATH=/root/mahjong
rm -rf base; cp -r botA base; mkdir -p base/data; ln -sf /root/mahjong/data/cnn_lad_chunjiandu.pkl base/data/cnn.pkl
JOBS=6; sem(){ while [ "$(jobs -rp|wc -l)" -ge "$JOBS" ]; do sleep 5; done; }
for c in 0.5_chunjiandu_full 0.7_chunjiandu_full 0.5_mythos_full 0.5_typec_full 0.5_scale_2000; do
  d="cf_$c"; rm -rf "$d"; cp -r botA "$d"; mkdir -p "$d/data"; ln -sf "/root/mahjong/cand_$c.pkl" "$d/data/cnn.pkl"
  for sd in 620000 621000 622000 623000 624000 625000; do
    out="res/cf_${c}_${sd}.log"; [ -f "$out" ] && grep -q requested, "$out" && continue
    sem; ( WALL_SEED_BASE=$sd timeout 6000 python3 eval/bench_vs_bot.py "cd /root/mahjong/$d && $BC" "cd /root/mahjong/base && $BC" 144 "c_$c" opp > "$out" 2>&1 ) &
  done
done
wait; echo "CGAUNT_DONE $(date -u)" >> res/CGAUNT.log
