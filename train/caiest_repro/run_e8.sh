#!/bin/bash
# E8 harness: value-guided 1-ply vs base (cnn_lad), duplicate gate.
# 3 wall-seed blocks x lambda sweep. Free-guarded (skip if GPU0 busy is irrelevant: CPU only;
# but guard disk + honor STOP file). Writes per-cell JSON into e8_cells/.
set -u
cd /root/IJCAI-mahjong/train/caiest_repro
mkdir -p e8_cells
CAND=/root/assets/cnn_lad_chunjiandu.npz
VALUE=ckpt/value_256x40.pkl
WORKERS=80
SEEDS=400
LAMS="0 0.5 1.0 2.0 4.0 8.0"
# 3 blocks with disjoint seed ranges
BLOCKS="70000 80000 90000"
LOG=/root/IJCAI-mahjong/train/caiest_repro/e8_run.log
echo "=== E8 start $(date) ===" >> $LOG
for lam in $LAMS; do
  for s0 in $BLOCKS; do
    if [ -f /root/STOP_E8 ]; then echo "STOP_E8 present, halting $(date)" >> $LOG; exit 0; fi
    # disk guard: bail if <2G free
    FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    if [ "$FREE" -lt 2 ]; then echo "DISK LOW ${FREE}G, halting $(date)" >> $LOG; exit 0; fi
    OUT=e8_cells/lam${lam}_s${s0}.json
    if [ -f "$OUT" ]; then echo "skip existing $OUT" >> $LOG; continue; fi
    echo "--- lam=$lam s0=$s0 $(date) ---" >> $LOG
    python3 e8_gate.py --cand $CAND --ref $CAND --value $VALUE \
        --lam $lam --topk 5 --seeds $SEEDS --workers $WORKERS --seed0 $s0 \
        --out $OUT >> $LOG 2>&1
    echo "done lam=$lam s0=$s0 rc=$? $(date)" >> $LOG
  done
done
echo "=== E8 all cells done $(date) ===" >> $LOG
python3 e8_aggregate.py >> $LOG 2>&1
echo "=== E8 aggregate done $(date) ===" >> $LOG
