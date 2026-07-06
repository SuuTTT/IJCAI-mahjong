#!/bin/bash
# recipe_train.sh — RECIPE sweep GPU dispatcher. Trains 128x40 nets per recipe_queue.txt, 1 job/GPU.
# free-guard: only launch on a truly-idle GPU (mem<1500MB); a running 128x40 job uses ~2.5-4GB so it
# reads busy -> coexists with big-net jobs and never double-books. After launching, CONFIRMATION-WAIT:
# block until that GPU actually shows the allocation (mem>=1500) or the job dies, before scanning again
# (closes the startup race where data-load >55s left the GPU reading free). disk-guard(<2000MB abort),
# /root/STOP_RECIPE honored, setsid. Drops each .bn.pkl once its fused .pkl exists (disk hygiene).
cd /root/IJCAI-mahjong/train/caiest_repro || exit 1
LOG=/root/recipe_train.log; mkdir -p ckpt/recipe
QUEUE=recipe_queue.txt
BASE="--channels 128 --blocks 40 --seed 0"
END=$(( $(date +%s) + 39600 ))   # 11h cap
echo "$(date -u) recipe_train START" >> "$LOG"
gpu_mem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null|tr -d ' '; }
gpu_free(){ local m=$(gpu_mem "$1"); [ -n "$m" ] && [ "$m" -lt 1500 ]; }
disk_mb(){ df -m /root|awk 'NR==2{print $4}'; }
training(){ pgrep -f "e11_train.py.*ckpt/recipe/${1}.pkl" >/dev/null 2>&1; }

while [ "$(date +%s)" -lt "$END" ] && [ ! -f /root/STOP_RECIPE ]; do
  # disk hygiene: drop .bn.pkl whose fused .pkl exists
  for f in ckpt/recipe/*.pkl; do
    [ -f "$f" ] || continue; case "$f" in *.bn.pkl) continue;; esac
    bn="${f%.pkl}.bn.pkl"; [ -f "$bn" ] && rm -f "$bn" && echo "$(date -u) rmbn $bn" >> "$LOG"
  done
  [ "$(disk_mb)" -lt 2000 ] && { echo "$(date -u) DISK<2000MB ABORT" >> "$LOG"; break; }

  # pick next undone, not-currently-training config
  next_tag=""; next_flags=""
  while IFS='|' read -r tag flags; do
    tag="$(echo "$tag"|tr -d ' ')"
    [ -z "$tag" ] && continue
    case "$tag" in \#*) continue;; esac
    [ -f "ckpt/recipe/${tag}.pkl" ] && continue
    training "$tag" && continue
    next_tag="$tag"; next_flags="$flags"; break
  done < "$QUEUE"

  if [ -z "$next_tag" ]; then
    if ! pgrep -f "e11_train.py.*ckpt/recipe/" >/dev/null 2>&1; then
      echo "$(date -u) queue drained + no jobs alive, exit" >> "$LOG"; break
    fi
    sleep 120; continue
  fi

  # find a truly-idle GPU
  launched=0
  for g in 0 1 2 3; do
    if gpu_free "$g" && [ "$(disk_mb)" -ge 1500 ]; then
      out="ckpt/recipe/${next_tag}.pkl"
      CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=6 setsid python3 e11_train.py \
        $BASE $next_flags --out "$out" >> "$LOG" 2>&1 < /dev/null &
      newpid=$!
      echo "$(date -u) GPU$g START $next_tag flags=[$next_flags] pid=$newpid" >> "$LOG"
      # CONFIRMATION-WAIT: block until GPU g shows allocation (mem>=1500) or job dies, <=300s
      for w in $(seq 1 60); do
        sleep 5
        kill -0 "$newpid" 2>/dev/null || { echo "$(date -u) GPU$g $next_tag died during startup" >> "$LOG"; break; }
        mm=$(gpu_mem "$g")
        [ -n "$mm" ] && [ "$mm" -ge 1500 ] && { echo "$(date -u) GPU$g $next_tag allocated ${mm}MB" >> "$LOG"; break; }
      done
      launched=1; break
    fi
  done
  [ "$launched" -eq 0 ] && sleep 60
done
echo "$(date -u) recipe_train END" >> "$LOG"
