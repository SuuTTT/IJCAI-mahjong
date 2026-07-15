#!/bin/bash
# Noisy-H1 replication: train 18 students distilled from all 8 noisy teachers,
# packing onto free GPUs (mem<2000MB, <=3/GPU). For CI on distill-vs-seed gap.
cd /root/crossgame/doudizhu
mkdir -p ckpt/students_repl logs
T8=$(ls ckpt/teachers_noisy/dou_nteacher_s20*.pkl | tr '\n' ',' | sed 's/,$//')
pick_gpu() {
  for g in 0 1 2 3 4 5 6 7; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null)
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null | wc -l)
    if [ "${m:-99999}" -lt 14000 ] && [ "${n:-9}" -lt 3 ]; then echo $g; return; fi
  done
  echo -1
}
for seed in $(seq 5000 5017); do
  out=ckpt/students_repl/dou_rstudent_s${seed}.pkl
  [ -f "$out" ] && continue
  while true; do
    g=$(pick_gpu)
    [ "$g" != "-1" ] && break
    sleep 30
  done
  CUDA_VISIBLE_DEVICES=$g nohup python3 dou_kd_train.py --data dou_data_noisy.npz --teachers $T8 --seed $seed --steps 25000 --alpha 0.7 --out $out > logs/rstudent_s${seed}.log 2>&1 &
  sleep 25
done
wait
echo REPL_STUDENTS_DONE
