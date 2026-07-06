#!/bin/bash
# E4 AWR training harness: beta-sweep x 2 seeds, KL-leash (lr 5e-5, mix 0.3), 8000 steps.
# 1 job/GPU, free-guard (skip GPU if mem>800MB), honor /root/STOP_E4.
cd /root/IJCAI-mahjong/train/caiest_repro
LOGD=logs/e4
OUTD=ckpt/e4
MOYU=/root/assets/moyu_bn_128x40.pkl
STEPS=8000
LR=5e-5
MIX=0.3

# job list: "beta seed"
JOBS=()
for b in 0 0.5 1.0 2.0 5.0; do
  for s in 1 2; do
    JOBS+=("$b $s")
  done
done

free_gpu() {  # echo index of a free GPU (mem<800MB) not currently assigned, else nothing
  for g in 0 1 2 3; do
    [ -n "${BUSY[$g]}" ] && continue
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$m" -lt 800 ]; then echo $g; return; fi
  done
}

declare -A BUSY      # gpu -> pid
declare -A JOBOF     # gpu -> jobdesc
qi=0
echo "E4 train start $(date) — ${#JOBS[@]} jobs" | tee -a $LOGD/driver.log

while [ $qi -lt ${#JOBS[@]} ] || [ ${#BUSY[@]} -gt 0 ]; do
  if [ -f /root/STOP_E4 ]; then echo "STOP_E4 seen, halting launches" | tee -a $LOGD/driver.log; break; fi
  # reap finished
  for g in "${!BUSY[@]}"; do
    pid=${BUSY[$g]}
    if ! kill -0 $pid 2>/dev/null; then
      echo "$(date) GPU$g done: ${JOBOF[$g]} (pid $pid)" | tee -a $LOGD/driver.log
      unset BUSY[$g]; unset JOBOF[$g]
    fi
  done
  # launch on free GPUs
  if [ $qi -lt ${#JOBS[@]} ]; then
    g=$(free_gpu)
    if [ -n "$g" ]; then
      read beta seed <<< "${JOBS[$qi]}"
      tag="b${beta}_s${seed}"
      out=$OUTD/awr_${tag}.pkl
      log=$LOGD/${tag}.log
      echo "$(date) launch GPU$g: beta=$beta seed=$seed -> $out" | tee -a $LOGD/driver.log
      nohup python3 awr_critic.py --init $MOYU --channels 128 --blocks 40 \
        --steps $STEPS --lr $LR --mix $MIX --beta $beta --wlo 0.0 --whi 10.0 \
        --seed $seed --gpu $g --out $out > $log 2>&1 &
      BUSY[$g]=$!; JOBOF[$g]="$tag"
      qi=$((qi+1))
      sleep 8
    fi
  fi
  sleep 5
done
echo "E4 train ALL_DONE $(date)" | tee -a $LOGD/driver.log
touch /root/IJCAI-mahjong/train/caiest_repro/logs/e4/TRAIN_DONE
