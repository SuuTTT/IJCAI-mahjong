#!/bin/bash
# TEMPORAL-PUSH gate queue: smoke (20 seeds) then 12 blocks x 2000 games per variant vs aug_s0
cd /root/IJCAI-mahjong/train/caiest_repro
REF=ckpt/aug/aug_128x40_s0.pkl
G=ckpt/archx/gates
declare -A KIND CFG SEED0
KIND[temporal_biggru_s12]=temporal;   CFG[temporal_biggru_s12]="channels=128,blocks=40,emb=64,gru=512,gru_layers=2";  SEED0[temporal_biggru_s12]=940000
KIND[temporal_tf_s13]=transformer;    CFG[temporal_tf_s13]="channels=128,blocks=40,emb=64,heads=8,tf_layers=4";       SEED0[temporal_tf_s13]=950000
KIND[temporal_strongaug_s11]=temporal;CFG[temporal_strongaug_s11]="channels=128,blocks=40,emb=64,gru=256";            SEED0[temporal_strongaug_s11]=960000
KIND[temporal_combo_s14]=temporal;    CFG[temporal_combo_s14]="channels=128,blocks=40,emb=64,gru=384";                SEED0[temporal_combo_s14]=970000
for V in temporal_biggru_s12 temporal_tf_s13 temporal_strongaug_s11 temporal_combo_s14; do
  # smoke: 20 seeds; abort variant on failure
  python3 gate_seq.py --cand ckpt/archx/$V.pkl --cand-kind ${KIND[$V]} --cand-cfg "${CFG[$V]}" \
    --ref $REF --seeds 20 --workers 20 --seed0 99999000 --out /tmp/smoke_$V.json \
    || { echo "SMOKE FAIL $V" >> temporal_orch.log; continue; }
  for i in $(seq 0 11); do
    S=$(( ${SEED0[$V]} + i*1000 ))
    [ -f $G/${V}_s$S.json ] && continue
    while [ "$(awk "{print (\$1>100)?1:0}" /proc/loadavg)" = "1" ]; do sleep 60; done
    python3 gate_seq.py --cand ckpt/archx/$V.pkl --cand-kind ${KIND[$V]} --cand-cfg "${CFG[$V]}" \
      --ref $REF --seeds 500 --workers 56 --seed0 $S --out $G/${V}_s$S.json
  done
done
python3 agg_temporal.py > TEMPORAL_PUSH_RESULTS.txt 2>&1
echo "TEMPORAL ORCH ALL DONE" >> temporal_orch.log
