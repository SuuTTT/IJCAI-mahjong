#!/bin/bash
# rl_online_harness.sh — durable driver for ONLINE self-play RL (rl_online.py) + calibrated
# multi-block gating of each snapshot vs aug_s0. Survives session drop (setsid). Honors
# /root/STOP_RL and a 15GB free-disk guard. Coexists with the recipe sweep + arch experiment
# (CPU-capped actors; gate uses few workers; grabs GPU sparingly).
cd /root/IJCAI-mahjong/train/caiest_repro || exit 1
LOG=/root/rl_online.harness.log
SNAPDIR=ckpt/rl_online
BLOCKDIR=ckpt/rl_online/gates
REF=ckpt/aug/aug_128x40_s0.pkl
mkdir -p "$SNAPDIR" "$BLOCKDIR"
echo "$(date -u) rl_online_harness START pid=$$" >> "$LOG"

disk_free_mb(){ df -m /root | awk 'NR==2{print $4}'; }
train_alive(){ pgrep -f "rl_online.py --tag $TAG" >/dev/null 2>&1; }

TAG=rlon
# ---- launch the trainer (background; it self-checkpoints snapshots) ----
if ! train_alive; then
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 nohup python3 -u rl_online.py \
    --tag "$TAG" --actors 24 --games-per-actor 2 --iters 6000 \
    --lr 2e-5 --beta-kl 0.4 --ent 0.008 --p-anchor 0.6 --snap-every 25 \
    --snapdir "$SNAPDIR" --minutes 1200 >> /root/rl_online.train.log 2>&1 < /dev/null &
  echo "$(date -u) launched trainer pid=$!" >> "$LOG"
fi

sleep 20
# ---- gating loop: gate each new snapshot at 6 blocks x 150 seeds vs aug_s0 ----
while : ; do
  [ -f /root/STOP_RL ] && { echo "$(date -u) STOP_RL -> harness exit" >> "$LOG"; break; }
  FREE=$(disk_free_mb)
  if [ -n "$FREE" ] && [ "$FREE" -lt 15000 ]; then
    # prune older gated fused snapshots to reclaim space (keep last 12 + final)
    ls -1t "$SNAPDIR"/snap_*.pkl 2>/dev/null | grep -v final | tail -n +13 | xargs -r rm -f
    echo "$(date -u) disk<15GB free=${FREE} pruned old snaps" >> "$LOG"
    FREE=$(disk_free_mb)
    [ "$FREE" -lt 12000 ] && { echo "$(date -u) disk<12GB HARD guard, pausing gate 300s" >> "$LOG"; sleep 300; continue; }
  fi
  # newest ungated snapshot
  SNAP=""
  for f in $(ls -1t "$SNAPDIR"/snap_*.pkl 2>/dev/null); do
    bt=$(basename "$f" .pkl)         # snap_<tag>
    st=${bt#snap_}
    [ -f "$BLOCKDIR/${st}.done" ] && continue
    SNAP="$f"; STAG="$st"; break
  done
  if [ -z "$SNAP" ]; then
    train_alive || { echo "$(date -u) trainer done + all gated -> harness exit" >> "$LOG"; break; }
    sleep 45; continue
  fi
  echo "$(date -u) gating $STAG" >> "$LOG"
  for k in 0 1 2 3 4 5; do
    S0=$((800000 + k*13000))
    python3 e11_gate.py --cand "$SNAP" --ref "$REF" \
      --cand-cfg channels=128,blocks=40 --ref-cfg channels=128,blocks=40 \
      --seeds 150 --workers 16 --seed0 "$S0" \
      --out "$BLOCKDIR/${STAG}_b${k}.json" >> /root/rl_online.gate.log 2>&1
    [ -f /root/STOP_RL ] && break
  done
  python3 rl_online_agg.py "$STAG" "$BLOCKDIR" >> "$LOG" 2>&1
  touch "$BLOCKDIR/${STAG}.done"
done
echo "$(date -u) rl_online_harness END" >> "$LOG"
