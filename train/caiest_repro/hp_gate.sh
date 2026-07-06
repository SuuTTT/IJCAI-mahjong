#!/bin/bash
cd /root/IJCAI-mahjong/train/caiest_repro
CAND=${1:-ckpt/archx/temporal_s0.pkl}
CCFG=${2:-channels=128,blocks=40,emb=64,gru=256}
TAG=${3:-temporal_s0}
REF=ckpt/aug/aug_128x40_s0.pkl
S0=${4:-906000}
N=${5:-12}
for i in $(seq 0 $((N-1))); do
  [ -f /root/STOP_TEMPORAL ] && { echo "STOP flag, abort"; exit 0; }
  SEED=$((S0 + i*1000))
  OUT=ckpt/archx/gates/${TAG}_s${SEED}.json
  [ -f "$OUT" ] && { echo "exists $OUT skip"; continue; }
  # wait until 1-min load < 18 before launching this block
  while :; do
    L=$(awk "{print int(\$1)}" /proc/loadavg)
    [ "$L" -lt 18 ] && break
    echo "load $L >=18 wait"; sleep 20
  done
  echo "=== block seed0=$SEED -> $OUT ($(date)) ==="
  python3 gate_seq.py --cand "$CAND" --cand-cfg "$CCFG" --ref "$REF" \
    --ref-kind resbn_fused --ref-cfg channels=128,blocks=40 \
    --seeds 500 --workers 24 --seed0 $SEED --out "$OUT" 2>&1
done
echo "HP_GATE_DONE $TAG"
