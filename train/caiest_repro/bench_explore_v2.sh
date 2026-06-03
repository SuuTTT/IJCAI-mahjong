#!/usr/bin/env bash
# Overnight monitor for SL arch search v2: poll boxes; as each finishes, pull + benchmark vs r18
# AND vs the resbn40 champ. Results -> /tmp/explore_v2_results.txt. Prunes nothing (small).
set -u
cd "$(dirname "$0")"; ROOT=$(cd ../.. && pwd)
KEY=~/.ssh/id_ed25519; SSHO="-i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes"
R18="MODEL=train/checkpoints/pbt_champion_fp16.npz OPENBLAS_NUM_THREADS=1 python3 bot/ml_bot.py"
RESBN40="BOTZONE_JSON=0 EXP_KIND=resbn EXP_CFG='{\"channels\":128,\"blocks\":40}' CAIEST_MODEL=$ROOT/train/caiest_repro/arch_ck/explore/resbn40.pkl PYTHONPATH=$ROOT/train/caiest_repro OPENBLAS_NUM_THREADS=1 python3 $ROOT/train/caiest_repro/explore_bot.py"
RES=/tmp/explore_v2_results.txt; : > "$RES"; mkdir -p arch_ck/explore
MAP=(
 "58.79.62.163 31161 resbn24   resbn    {\"channels\":128,\"blocks\":24}"
 "185.201.68.112 43153 resbn56  resbn    {\"channels\":128,\"blocks\":56}"
 "91.150.160.38 17204 resbnw192 resbn    {\"channels\":192,\"blocks\":24}"
 "154.61.62.158 50017 resbnattn cnn_attn {\"channels\":128,\"conv_blocks\":16,\"layers\":4,\"heads\":8}"
)
declare -A done_b
deadline=$(( $(date +%s) + 28800 ))   # 8h
while [ $(date +%s) -lt $deadline ]; do
  all=1
  for row in "${MAP[@]}"; do
    set -- $row; ip=$1; port=$2; tag=$3; kind=$4; cfg=$5
    [ "${done_b[$tag]:-0}" = 1 ] && continue
    line=$(timeout 25 ssh $SSHO -p $port root@$ip "tail -1 /root/mjx/$tag.log 2>/dev/null" 2>/dev/null)
    if echo "$line" | grep -q "^DONE"; then
      echo "[$tag] DONE: $line" | tee -a "$RES"
      rsync -az --timeout=180 -e "ssh $SSHO -p $port" root@$ip:/root/mjx/$tag.pkl arch_ck/explore/ 2>/dev/null
      BOT="BOTZONE_JSON=0 EXP_KIND=$kind EXP_CFG='$cfg' CAIEST_MODEL=$ROOT/train/caiest_repro/arch_ck/explore/$tag.pkl PYTHONPATH=$ROOT/train/caiest_repro OPENBLAS_NUM_THREADS=1 python3 $ROOT/train/caiest_repro/explore_bot.py"
      r1=$(cd "$ROOT" && OPENBLAS_NUM_THREADS=1 python3 eval/bench_vs_bot.py "$R18" "$BOT" 60 r18 "$tag" 2>&1 | grep -E "$tag:|draws=")
      r2=$(cd "$ROOT" && OPENBLAS_NUM_THREADS=1 python3 eval/bench_vs_bot.py "$RESBN40" "$BOT" 60 resbn40 "$tag" 2>&1 | grep -E "$tag:|draws=")
      { echo "  vs r18:    $r1"; echo "  vs resbn40:$r2"; echo ""; } | tee -a "$RES"
      done_b[$tag]=1
    else
      all=0
    fi
  done
  [ $all = 1 ] && { echo "=== ALL v2 BENCHMARKED ===" | tee -a "$RES"; break; }
  sleep 180
done
echo "===== FINAL v2 =====" >> "$RES"; cat "$RES"
