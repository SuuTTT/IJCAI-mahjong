#!/bin/bash
cd /root/IJCAI-mahjong/train/caiest_repro || exit 1
mkdir -p ckpt/bn384
LOG=ckpt/bn384/confirm.log
W=32
NS=1000
DISTILL=/root/assets/cnn_lad_chunjiandu.npz
echo "=== START $(date) pid=$$ workers=$W seeds/block=$NS ===" >> "$LOG"

run_block () {  # $1=out $2=cand $3=cand-kind $4=cand-cfg $5=seed0
  [ -f /root/STOP_BN384 ] && { echo "STOP_BN384 present, halting $(date)" >> "$LOG"; return 1; }
  AVAIL=$(df -P / | awk "NR==2{print \$4}")   # KB free
  if [ "$AVAIL" -lt 1500000 ]; then echo "DISK LOW ($AVAIL KB), halting $(date)" >> "$LOG"; return 1; fi
  [ -f "$1" ] && { echo "skip exists $1" >> "$LOG"; return 0; }
  echo "RUN $(basename $1) seed0=$5 $(date)" >> "$LOG"
  python3 e8_gate.py --cand "$2" --cand-kind "$3" --cand-cfg "$4" \
    --ref "$DISTILL" --ref-kind resbn_fused --ref-cfg channels=128,blocks=40 \
    --lam 0 --seeds $NS --workers $W --seed0 $5 --out "$1" >> "$LOG" 2>&1
  return 0
}

# 1) PRIORITY: full_384x40_s0 vs distill, 12 disjoint blocks
for S in 300000 301000 302000 303000 304000 305000 306000 307000 308000 309000 310000 311000; do
  run_block ckpt/bn384/full_384x40_s0_s${S}.json ckpt/e1b/full_384x40_s0.pkl resbn_fused channels=384,blocks=40 $S || break
done
python3 bn384_confirm_agg.py >> "$LOG" 2>&1; echo "--- partial agg after candidate $(date) ---" >> "$LOG"

# 2) Calibration: distill vs distill must read 2.500, 4 blocks
for S in 400000 401000 402000 403000; do
  run_block ckpt/bn384/calib_distill_s${S}.json "$DISTILL" resbn_fused channels=128,blocks=40 $S || break
done
python3 bn384_confirm_agg.py >> "$LOG" 2>&1; echo "--- agg after calib $(date) ---" >> "$LOG"

# 3) Generality: big192x40_s0_fused vs distill, 8 blocks
for S in 500000 501000 502000 503000 504000 505000 506000 507000; do
  run_block ckpt/bn384/big192x40_s0_fused_s${S}.json ckpt/big192x40_s0_fused.pkl resbn_fused channels=192,blocks=40 $S || break
done

python3 bn384_confirm_agg.py >> "$LOG" 2>&1
echo "=== ALL DONE $(date) ===" >> "$LOG"
