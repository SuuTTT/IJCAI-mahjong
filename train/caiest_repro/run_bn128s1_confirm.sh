#!/bin/bash
set -e
cd /root/IJCAI-mahjong/train/caiest_repro
CAND=ckpt/e1b/full_128x40_s1.pkl
REF=/root/assets/cnn_lad_chunjiandu.npz
OUTD=ckpt/bn128s1
for s in 400000 401000 402000 403000 404000 405000 406000 407000 408000 409000; do
  echo "=== block seed0=$s $(date) ==="
  python3 e8_gate.py --cand $CAND --cand-kind resbn_fused --cand-cfg channels=128,blocks=40 \
    --ref $REF --ref-kind resbn_fused --ref-cfg channels=128,blocks=40 \
    --lam 0 --topk 5 --seeds 500 --workers 40 --seed0 $s \
    --out $OUTD/full_128x40_s1_s${s}.json
done
echo "=== ALL BLOCKS DONE $(date) ==="
