#!/bin/bash
# run_bestnet_gate.sh — TLE + val + calibrated placement gate (vs aug_s0) for the BESTNET campaign.
# Gates raw big nets immediately, waits for the enhanced nets, re-aggregates after each.
# disk-guard(<2000MB), /root/STOP_BESTNET. setsid. CPU workers only (training holds the GPUs).
cd /root/IJCAI-mahjong/train/caiest_repro || exit 1
LOG=/root/bestnet_gate.log; mkdir -p ckpt/best/gates ckpt/best/tle ckpt/best/val
AUG=ckpt/aug/aug_128x40_s0.pkl; RCFG="channels=128,blocks=40"
SEEDS=${SEEDS:-500}; WORKERS=${WORKERS:-64}; BLOCKS=${BLOCKS:-12}; BASE=700000
disk_mb(){ df -m /root|awk "NR==2{print \$4}"; }
stop(){ [ -f /root/STOP_BESTNET ] || [ "$(disk_mb)" -lt 2000 ]; }
echo "$(date -u) run_bestnet_gate START seeds=$SEEDS workers=$WORKERS blocks=$BLOCKS" >> "$LOG"

tle(){ local tag=$1 ck=$2 ch=$3 blk=$4; [ -f "ckpt/best/tle/${tag}.json" ] && return
  OMP_NUM_THREADS=1 python3 bestnet_time.py "$ck" "$ch" "$blk" > "ckpt/best/tle/${tag}.json" 2>>"$LOG"
  echo "$(date -u) tle $tag rc=$?" >> "$LOG"; }
valacc(){ local tag=$1 ck=$2 ch=$3 blk=$4; [ -f "ckpt/best/val/${tag}.json" ] && return
  python3 bestnet_val.py "$ck" "$ch" "$blk" "$tag" >> "$LOG" 2>&1; echo "$(date -u) val $tag rc=$?" >> "$LOG"; }
gate(){ local tag=$1 cand=$2 ccfg=$3
  for b in $(seq 0 $((BLOCKS-1))); do
    stop && { echo "$(date -u) STOP/DISK halt in $tag" >> "$LOG"; return; }
    local s0=$((BASE+b*1000)); local out="ckpt/best/gates/${tag}_s${s0}.json"
    [ -f "$out" ] && continue
    python3 e11_gate.py --cand "$cand" --cand-cfg "$ccfg" --ref "$AUG" --ref-cfg "$RCFG" \
      --seeds "$SEEDS" --workers "$WORKERS" --seed0 "$s0" --out "$out" >> "$LOG" 2>&1
    echo "$(date -u) gated $tag block $b rc=$?" >> "$LOG"
  done; }

# 0) calibration: aug_s0 vs aug_s0 must = 2.500
[ -f ckpt/best/gates/calib_s${BASE}.json ] || python3 e11_gate.py --cand "$AUG" --cand-cfg "$RCFG" \
  --ref "$AUG" --ref-cfg "$RCFG" --seeds "$SEEDS" --workers "$WORKERS" --seed0 "$BASE" \
  --out ckpt/best/gates/calib_s${BASE}.json >> "$LOG" 2>&1
echo "$(date -u) calib done" >> "$LOG"

# 1) raw big nets (already trained) — TLE + val + gate immediately
tle raw384 ckpt/e1b/full_384x40_s0.pkl 384 40;  valacc raw384 ckpt/e1b/full_384x40_s0.pkl 384 40
tle raw192 ckpt/big192x40_s0_fused.pkl 192 40;  valacc raw192 ckpt/big192x40_s0_fused.pkl 192 40
gate raw384 ckpt/e1b/full_384x40_s0.pkl "channels=384,blocks=40"
python3 bestnet_agg.py >> "$LOG" 2>&1
gate raw192 ckpt/big192x40_s0_fused.pkl "channels=192,blocks=40"
python3 bestnet_agg.py >> "$LOG" 2>&1

# 2) enhanced nets — wait for each, then TLE+val+gate, re-aggregate
declare -A EN=( [enh384_s0]="ckpt/best/enh_384x40_s0.pkl 384 40" [enh384_s1]="ckpt/best/enh_384x40_s1.pkl 384 40" \
                [enh192_s0]="ckpt/best/enh_192x40_s0.pkl 192 40" [enh192_s1]="ckpt/best/enh_192x40_s1.pkl 192 40" )
for tag in enh384_s0 enh384_s1 enh192_s0 enh192_s1; do
  set -- ${EN[$tag]}; ck=$1; ch=$2; blk=$3
  while [ ! -f "$ck" ]; do stop && { echo "$(date -u) STOP before $tag" >> "$LOG"; break 2; }; sleep 120; done
  [ -f "$ck" ] || break
  sleep 20
  tle "$tag" "$ck" "$ch" "$blk"; valacc "$tag" "$ck" "$ch" "$blk"
  gate "$tag" "$ck" "channels=${ch},blocks=${blk}"
  python3 bestnet_agg.py >> "$LOG" 2>&1
  echo "$(date -u) FINISHED $tag" >> "$LOG"
done
python3 bestnet_agg.py >> "$LOG" 2>&1
echo "$(date -u) run_bestnet_gate DONE" >> "$LOG"
