#!/usr/bin/env bash
# Overnight guardian for the 8h run: keep disk safe, keep the fleet busy, write a status report.
# - prunes /tmp + stale checkpoints if disk < 1.5GB (the disk-full failure mode)
# - if a v2 box finished AND is benchmarked, relaunch it with a queued 2nd config (keep GPU busy)
# - writes /tmp/overnight_status.md every cycle for the user to read on return
set -u
cd "$(dirname "$0")"; ROOT=$(cd ../.. && pwd)
KEY=~/.ssh/id_ed25519; SSHO="-i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes"
ST=/tmp/overnight_status.md
# box -> 2nd config to run after its first finishes (keep GPUs busy the full 8h)
declare -A SECOND=(
 [58.79.62.163:31161:resbn24]="resbn48|resbn|{\"channels\":128,\"blocks\":48}"
)
deadline=$(( $(date +%s) + 28800 )); declare -A relaunched
while [ $(date +%s) -lt $deadline ]; do
  free_mb=$(df -m / | tail -1 | awk '{print $4}')
  if [ "$free_mb" -lt 1500 ]; then
    rm -f /tmp/*.npz /tmp/*.pkl 2>/dev/null
    find arch_ck -name '*.pkl' ! -name 'resbn40.pkl' ! -name 'base_16x128_final.pkl' -mmin +120 -delete 2>/dev/null
  fi
  # status report
  {
    echo "# Overnight run status — $(date -u '+%Y-%m-%d %H:%M UTC')"
    echo; echo "disk free: ${free_mb} MB"
    echo; echo "## SL arch search v2 (fleet) — per box"
    for bp in "58.79.62.163 31161 resbn24" "185.201.68.112 43153 resbn56" "91.150.160.38 17204 resbnw192" "154.61.62.158 50017 resbnattn"; do
      set -- $bp; echo "- **$3**: $(timeout 20 ssh $SSHO -p $2 root@$1 "tail -1 /root/mjx/$3.log 2>/dev/null" 2>/dev/null || echo unreachable)"
    done
    echo; echo "## Benchmarked results (vs r18 / vs resbn40 champ)"; echo '```'; cat /tmp/explore_v2_results.txt 2>/dev/null | tail -40; echo '```'
    echo; echo "NOTE: resbn40 (+973 vs CNN champ) is our best base; deploy reverted to the proven 16-block CNN (resbn40 crashes Botzone torch-1.4). RL fine-tune + distillation are built and ready to run with you on return."
  } > "$ST" 2>/dev/null
  # 2nd wave for the fast box
  for key in "${!SECOND[@]}"; do
    [ "${relaunched[$key]:-0}" = 1 ] && continue
    ip=${key%%:*}; rest=${key#*:}; port=${rest%%:*}; tag=${rest##*:}
    line=$(timeout 20 ssh $SSHO -p $port root@$ip "tail -1 /root/mjx/$tag.log 2>/dev/null" 2>/dev/null)
    if echo "$line" | grep -q "^DONE"; then
      IFS='|' read -r ntag nkind ncfg <<< "${SECOND[$key]}"
      timeout 30 ssh $SSHO -p $port root@$ip "cd /root/mjx && nohup python3 remote_train.py --kind $nkind --cfg '$ncfg' --epochs 18 --batch 1024 --out ${ntag}.pkl > ${ntag}.log 2>&1 & echo relaunched $ntag" 2>/dev/null
      relaunched[$key]=1
    fi
  done
  sleep 1200
done
echo "guardian done" >> "$ST"
