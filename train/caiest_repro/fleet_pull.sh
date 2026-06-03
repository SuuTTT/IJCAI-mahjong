#!/usr/bin/env bash
# Poll the fleet SL-exploration boxes; when each finishes, pull its .pkl + print final val_acc.
# Run from train/caiest_repro/. Writes pulled models to arch_ck/explore/.
set -u
KEY=~/.ssh/id_ed25519
SSHO="-i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes"
mkdir -p arch_ck/explore
# ip port tag
BOXES="47.186.29.91:52734:attn 58.79.62.163:31161:cnnattn 154.61.62.158:50017:resbn40 185.201.68.112:43153:gnn 91.150.160.38:17204:attnbig"
deadline=$(( $(date +%s) + 9000 ))
declare -A done
while [ $(date +%s) -lt $deadline ]; do
  alldone=1
  for b in $BOXES; do
    ip=${b%%:*}; rest=${b#*:}; port=${rest%%:*}; tag=${rest##*:}
    [ "${done[$tag]:-0}" = 1 ] && continue
    line=$(timeout 25 ssh $SSHO -p $port root@$ip "tail -1 /root/mjx/${tag}.log 2>/dev/null" 2>/dev/null)
    if echo "$line" | grep -q "^DONE"; then
      echo "[$tag] $line"
      rsync -az --timeout=120 -e "ssh $SSHO -p $port" root@$ip:/root/mjx/${tag}.pkl arch_ck/explore/ 2>/dev/null && echo "[$tag] pulled -> arch_ck/explore/${tag}.pkl"
      done[$tag]=1
    else
      alldone=0
      echo "[$tag] $line"
    fi
  done
  [ $alldone = 1 ] && { echo "=== ALL DONE ==="; break; }
  sleep 120
done
echo "=== pulled models ==="; ls -la arch_ck/explore/ 2>/dev/null
