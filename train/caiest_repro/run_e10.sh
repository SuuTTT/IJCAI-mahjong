#!/bin/bash
# E10 harness: RISK-SEEKING value-guided 1-ply vs base (cnn_lad), duplicate gate.
# Primary sweep (mu=0): lam in {0 0.5 1 2 4 8 16}. Guarded arm: lam=8 with mu in {8 16}.
# 3 wall-seed blocks (70000/80000/90000). CPU-only; E1b owns GPU. Free-guarded: STOP_E10 + disk.
set -u
cd /root/IJCAI-mahjong/train/caiest_repro
mkdir -p e10_cells
CAND=/root/assets/cnn_lad_chunjiandu.npz
VALUE=ckpt/value_256x40.pkl
WORKERS=80
SEEDS=400
BLOCKS="70000 80000 90000"
LOG=/root/IJCAI-mahjong/train/caiest_repro/e10_run.log
echo "=== E10 start $(date) ===" >> $LOG

run_cell () {  # args: lam mu
  local lam=$1 mu=$2
  for s0 in $BLOCKS; do
    if [ -f /root/STOP_E10 ]; then echo "STOP_E10 present, halting $(date)" >> $LOG; exit 0; fi
    FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    if [ "$FREE" -lt 2 ]; then echo "DISK LOW ${FREE}G, halting $(date)" >> $LOG; exit 0; fi
    OUT=e10_cells/lam${lam}_mu${mu}_s${s0}.json
    if [ -f "$OUT" ]; then echo "skip existing $OUT" >> $LOG; continue; fi
    echo "--- lam=$lam mu=$mu s0=$s0 $(date) ---" >> $LOG
    python3 e10_gate.py --cand $CAND --ref $CAND --value $VALUE \
        --lam $lam --mu $mu --topk 5 --seeds $SEEDS --workers $WORKERS --seed0 $s0 \
        --out $OUT >> $LOG 2>&1
    echo "done lam=$lam mu=$mu s0=$s0 rc=$? $(date)" >> $LOG
  done
}

# primary risk-seeking sweep (mu=0)
for lam in 0 0.5 1 2 4 8 16; do run_cell $lam 0; done
# V_4th-guarded upside arm at the lambda where override actually bites
for mu in 8 16; do run_cell 8 $mu; done

echo "=== E10 all cells done $(date) ===" >> $LOG
python3 e10_aggregate.py >> $LOG 2>&1
echo "=== E10 aggregate done $(date) ===" >> $LOG
