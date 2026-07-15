#!/bin/bash
# E1 second half: CIFAR-100N (clean control + noisy fine labels ~40%).
# GPU picker: >8GB free AND no compute process (excludes the 10N tail,
# the value-head retrain, and anything else running). Never kills anything.
cd /root/e1_cifarn || exit 1
mkdir -p logs results ckpt
LEVELS=(clean noisy)

BUSY=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | sort -u | tr '\n' ' ')
mapfile -t FREE < <(nvidia-smi --query-gpu=index,pci.bus_id,memory.free --format=csv,noheader,nounits \
  | awk -F', ' -v busy="$BUSY" '($3 > 8000) && index(busy, $2) == 0 {print $1}')
n=${#FREE[@]}
if [ "$n" -eq 0 ]; then echo "[run_e1_c100] NO idle GPU with >8GB free, aborting"; exit 1; fi
[ "$n" -gt 2 ] && n=2
echo "[run_e1_c100] idle GPUs picked: ${FREE[*]:0:$n} ($(date))"

for ((g=0; g<n; g++)); do
  (
    for ((i=g; i<2; i+=n)); do
      lv=${LEVELS[$i]}
      echo "[run_e1_c100] level c100_$lv -> GPU ${FREE[$g]} start $(date)"
      python3 cifarn_e1.py --dataset c100 --noise "$lv" --gpu "${FREE[$g]}" --epochs 60 --data data >> "logs/c100_${lv}.log" 2>&1
      echo "[run_e1_c100] level c100_$lv exit $? $(date)"
    done
  ) &
done
wait
python3 aggregate.py > logs/aggregate_c100.log 2>&1
echo "[run_e1_c100] ALL DONE $(date)"
touch results/E1_C100_DONE
