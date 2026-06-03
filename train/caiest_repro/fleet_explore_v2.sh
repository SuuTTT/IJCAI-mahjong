#!/usr/bin/env bash
# SL arch search v2 — build on resbn40's win (normalized depth). Explore deeper/wider/hybrid
# resbn variants across free GPUs. Run from train/caiest_repro/. Pull+benchmark via bench v2.
set -u
KEY=~/.ssh/id_ed25519
SSHO="-i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes -o ServerAliveInterval=30"
EPOCHS=18
# ip port tag kind cfg   (all use existing models_explore archs)
read -r -d '' BOXES <<'EOF'
58.79.62.163 31161 resbn24   resbn    {"channels":128,"blocks":24}
185.201.68.112 43153 resbn56  resbn    {"channels":128,"blocks":56}
91.150.160.38 17204 resbnw192 resbn    {"channels":192,"blocks":24}
154.61.62.158 50017 resbnattn cnn_attn {"channels":128,"conv_blocks":16,"layers":4,"heads":8}
EOF
launch() {
  local ip=$1 port=$2 tag=$3 kind=$4 cfg=$5
  echo "[$tag] -> $ip:$port ($kind)"
  ssh $SSHO -p "$port" root@"$ip" "mkdir -p /root/mjx" 2>/dev/null || { echo "[$tag] SSH FAIL"; return; }
  rsync -az --timeout=180 -e "ssh $SSHO -p $port" \
     data/cooked_single.npz models_explore.py remote_train.py model.py feature.py agent.py \
     root@"$ip":/root/mjx/ 2>&1 | tail -1
  ssh $SSHO -p "$port" root@"$ip" \
     "cd /root/mjx && nohup python3 remote_train.py --kind $kind --cfg '$cfg' --epochs $EPOCHS --batch 1024 --out ${tag}.pkl > ${tag}.log 2>&1 & echo [$tag] LAUNCHED \$!"
}
echo "$BOXES" | while read -r ip port tag kind cfg; do
  [ -z "$ip" ] && continue
  launch "$ip" "$port" "$tag" "$kind" "$cfg" &
done
wait
echo "=== v2 launches dispatched ==="
