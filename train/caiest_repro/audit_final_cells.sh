#!/bin/bash
# audit_final_cells.sh — JOB 2: complete the papers' integrity/audit table.
# Fresh-wall cells for a03 (alpha=0.3) and the teacher-count curve points
# kd1t / kd2t. Ckpts pulled from HF Dannibal/ijcai-mahjong-ckpts-2026 (local
# copies were deleted). 12 disjoint blocks x 2000 seeds per cell, ensemble
# (3 ckpts, deploy mean-softmax rule) vs aug_s0 ref via e12_ens_gate.py.
# Cell seed0 bases 1000000 / 1030000 / 1060000, block i at base + i*2000
# (12*2000 = 24000 < 30000 => all 36 blocks disjoint).
# Idempotent: skips existing block JSONs. CPU-only, nice -n 10, 40 workers.
set -u
cd /root/caiest_repro || exit 1
mkdir -p ckpt/paperx ckpt/kdcurve audit_final results logs
BASE=https://huggingface.co/Dannibal/ijcai-mahjong-ckpts-2026/resolve/main
FILES="ckpt/paperx/a03_s0.pkl ckpt/paperx/a03_s1.pkl ckpt/paperx/a03_s2.pkl \
ckpt/kdcurve/kd1t_s0.pkl ckpt/kdcurve/kd1t_s1.pkl ckpt/kdcurve/kd1t_s3.pkl \
ckpt/kdcurve/kd2t_s0.pkl ckpt/kdcurve/kd2t_s1.pkl ckpt/kdcurve/kd2t_s2.pkl"

echo "[audit_final] start $(date -u)"
for f in $FILES; do
    ok=0
    for try in 1 2 3 4 5; do
        [ -s "$f" ] && { ok=1; break; }
        wget -q -O "$f.part" "$BASE/$f" && mv "$f.part" "$f" && { ok=1; break; }
        echo "[audit_final] wget retry $try for $f"; sleep 20
    done
    [ "$ok" = 1 ] || { echo "[audit_final] FATAL: could not fetch $f"; exit 2; }
done
echo "[audit_final] downloads done $(date -u)"

# verify every cand ckpt (and the ref) loads into the gate arch before burning CPU
python3 - <<'EOF' || exit 3
import torch, sys
sys.path.insert(0, '/root/caiest_repro')
from models_explore import build
paths = ("ckpt/aug/aug_128x40_s0.pkl "
         "ckpt/paperx/a03_s0.pkl ckpt/paperx/a03_s1.pkl ckpt/paperx/a03_s2.pkl "
         "ckpt/kdcurve/kd1t_s0.pkl ckpt/kdcurve/kd1t_s1.pkl ckpt/kdcurve/kd1t_s3.pkl "
         "ckpt/kdcurve/kd2t_s0.pkl ckpt/kdcurve/kd2t_s1.pkl ckpt/kdcurve/kd2t_s2.pkl").split()
import os
os.chdir('/root/caiest_repro')
for p in paths:
    m = build('resbn_fused', channels=128, blocks=40)
    m.load_state_dict(torch.load(p, map_location='cpu'))
    print('OK', p, flush=True)
EOF
echo "[audit_final] ckpt load-check passed $(date -u)"

A=ckpt/aug/aug_128x40_s0.pkl
cell() {  # cell <name> <cand-csv> <seed0-base>
    local NAME=$1 CAND=$2 B=$3 i
    for i in $(seq 0 11); do
        [ -f "audit_final/${NAME}_b$i.json" ] && continue
        echo "[audit_final] $NAME block $i seed0 $((B + i*2000)) $(date -u)"
        nice -n 10 python3 e12_ens_gate.py --cand "$CAND" --ref $A \
            --seeds 2000 --workers 40 --seed0 $((B + i*2000)) \
            --out "audit_final/${NAME}_b$i.json" \
            >> "logs/audit_${NAME}.log" 2>&1 \
            || echo "[audit_final] WARN: $NAME block $i failed"
    done
}
cell a03ens ckpt/paperx/a03_s0.pkl,ckpt/paperx/a03_s1.pkl,ckpt/paperx/a03_s2.pkl 1000000
cell kd1tens ckpt/kdcurve/kd1t_s0.pkl,ckpt/kdcurve/kd1t_s1.pkl,ckpt/kdcurve/kd1t_s3.pkl 1030000
cell kd2tens ckpt/kdcurve/kd2t_s0.pkl,ckpt/kdcurve/kd2t_s1.pkl,ckpt/kdcurve/kd2t_s2.pkl 1060000

python3 agg_audit_final.py && touch results/AUDIT_FINAL_DONE
echo "[audit_final] ALL DONE $(date -u)"
