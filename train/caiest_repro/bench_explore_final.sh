#!/usr/bin/env bash
# Poll fleet SL boxes; as each finishes (DONE in log), pull its converged .pkl and benchmark it
# vs r18 through the judge. Run from train/caiest_repro/. Results -> /tmp/explore_results.txt
set -u
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
KEY=~/.ssh/id_ed25519
SSHO="-i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes"
R18="MODEL=train/checkpoints/pbt_champion_fp16.npz OPENBLAS_NUM_THREADS=1 python3 bot/ml_bot.py"
RES=/tmp/explore_results.txt; : > "$RES"
mkdir -p arch_ck/explore
# ip port tag kind cfg
MAP=(
 "47.186.29.91 52734 attn     attn     {\"d_model\":192,\"layers\":6,\"heads\":8}"
 "58.79.62.163 31161 cnnattn  cnn_attn {\"channels\":128,\"conv_blocks\":4,\"layers\":4,\"heads\":8}"
 "154.61.62.158 50017 resbn40 resbn    {\"channels\":128,\"blocks\":40}"
 "91.150.160.38 17204 attnbig attn     {\"d_model\":256,\"layers\":8,\"heads\":8}"
)
declare -A done_b
deadline=$(( $(date +%s) + 25200 ))   # 7h
while [ $(date +%s) -lt $deadline ]; do
  all=1
  for row in "${MAP[@]}"; do
    set -- $row; ip=$1; port=$2; tag=$3; kind=$4; cfg=$5
    [ "${done_b[$tag]:-0}" = 1 ] && continue
    line=$(timeout 25 ssh $SSHO -p $port root@$ip "tail -1 /root/mjx/$tag.log 2>/dev/null" 2>/dev/null)
    if echo "$line" | grep -q "^DONE"; then
      echo "[$tag] FINISHED: $line"
      rsync -az --timeout=180 -e "ssh $SSHO -p $port" root@$ip:/root/mjx/$tag.pkl arch_ck/explore/ 2>/dev/null
      BOT="BOTZONE_JSON=0 EXP_KIND=$kind EXP_CFG='$cfg' CAIEST_MODEL=$ROOT/train/caiest_repro/arch_ck/explore/$tag.pkl PYTHONPATH=$ROOT/train/caiest_repro OPENBLAS_NUM_THREADS=1 python3 $ROOT/train/caiest_repro/explore_bot.py"
      echo "[$tag] benchmarking vs r18 (60 games)..."
      out=$(cd "$ROOT" && OPENBLAS_NUM_THREADS=1 python3 eval/bench_vs_bot.py "$R18" "$BOT" 60 r18 "$tag" 2>&1 | grep -E "$tag:|draws=")
      echo "[$tag] $line" >> "$RES"; echo "$out" >> "$RES"; echo "" >> "$RES"
      echo "[$tag] RESULT: $out"
      done_b[$tag]=1
    else
      all=0; echo "[$tag] $line"
    fi
  done
  [ $all = 1 ] && { echo "=== ALL BENCHMARKED ==="; break; }
  sleep 180
done
echo "===== FINAL EXPLORE RESULTS ====="; cat "$RES"
