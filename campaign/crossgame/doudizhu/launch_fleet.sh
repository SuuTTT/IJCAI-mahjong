#!/bin/bash
# Launch 8 BC teacher trainings, one per seed, mapped to FREE GPUs (4-7, 2 each),
# staggered 30s. GPUs 0-3 run unrelated mahjong jobs and must NOT be touched.
cd /root/crossgame/doudizhu || exit 1
mkdir -p ckpt/teachers logs
GPUS=(4 5 6 7 4 5 6 7)
for i in 0 1 2 3 4 5 6 7; do
  g=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$g nohup python3 dou_bc_train.py \
      --seed $i --steps 60000 --lr 5e-4 --bs 512 \
      --out ckpt/teachers/dou_teacher_s${i}.pkl \
      > logs/teacher_s${i}.log 2>&1 &
  echo "launched teacher seed=$i on GPU=$g pid=$! at $(date +%T)"
  sleep 30
done
wait
echo "ALL TEACHERS EXITED at $(date +%T)"
