#!/bin/bash
# E1 CIFAR-10N launcher: memory-based GPU picker (>8GB free only), one noise
# level per free GPU (round-robin if fewer than 4 free). NEVER touches the
# e13 trainers (GPUs 4-7) or the e12 CPU gates.
cd /root/e1_cifarn || exit 1
mkdir -p logs results ckpt
LEVELS=(clean aggre rand1 worst)

mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > 8000 {print $1}')
n=${#FREE[@]}
if [ "$n" -eq 0 ]; then echo "[run_e1] NO GPU with >8GB free, aborting"; exit 1; fi
[ "$n" -gt 4 ] && n=4
echo "[run_e1] free GPUs picked: ${FREE[*]:0:$n} ($(date))"

for ((g=0; g<n; g++)); do
  (
    for ((i=g; i<4; i+=n)); do
      lv=${LEVELS[$i]}
      echo "[run_e1] level $lv -> GPU ${FREE[$g]} start $(date)"
      python3 cifarn_e1.py --noise "$lv" --gpu "${FREE[$g]}" --epochs 60 --data data >> "logs/${lv}.log" 2>&1
      echo "[run_e1] level $lv exit $? $(date)"
    done
  ) &
done
wait
python3 aggregate.py > logs/aggregate.log 2>&1
echo "[run_e1] ALL DONE $(date)"
touch results/E1_DONE
