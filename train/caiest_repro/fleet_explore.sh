#!/usr/bin/env bash
# Fan out SL architecture exploration across free fleet GPUs. Run from train/caiest_repro/.
# Each box trains one arch on cooked_single.npz; pull *.pkl back with fleet_pull.sh.
set -u
KEY=~/.ssh/id_ed25519
SSHO="-i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes -o ServerAliveInterval=30"
EPOCHS=14
# box assignments: ip port tag kind cfg
read -r -d '' BOXES <<'EOF'
47.186.29.91 52734 attn      attn      {"d_model":192,"layers":6,"heads":8}
58.79.62.163 31161 cnnattn   cnn_attn  {"channels":128,"conv_blocks":4,"layers":4,"heads":8}
154.61.62.158 50017 resbn40  resbn     {"channels":128,"blocks":40}
185.201.68.112 43153 gnn      gnn       {"hidden":256,"layers":4}
91.150.160.38 17204 attnbig  attn      {"d_model":256,"layers":8,"heads":8}
EOF

launch() {
  local ip=$1 port=$2 tag=$3 kind=$4 cfg=$5
  echo "[$tag] -> $ip:$port ($kind)"
  ssh $SSHO -p "$port" root@"$ip" "mkdir -p /root/mjx" 2>/dev/null || { echo "[$tag] SSH FAIL"; return; }
  rsync -az --timeout=120 -e "ssh $SSHO -p $port" \
     data/cooked_single.npz models_explore.py remote_train.py model.py feature.py agent.py \
     root@"$ip":/root/mjx/ 2>&1 | tail -1
  ssh $SSHO -p "$port" root@"$ip" \
     "cd /root/mjx && nohup python3 remote_train.py --kind $kind --cfg '$cfg' --epochs $EPOCHS --batch 1024 --out ${tag}.pkl > ${tag}.log 2>&1 & echo [$tag] LAUNCHED pid \$!"
}

echo "$BOXES" | while read -r ip port tag kind cfg; do
  [ -z "$ip" ] && continue
  launch "$ip" "$port" "$tag" "$kind" "$cfg" &
done
wait
echo "=== all launches dispatched ==="
