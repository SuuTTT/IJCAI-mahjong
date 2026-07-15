#!/bin/bash
cd /root/caiest_repro
A=ckpt/aug/aug_128x40_s0.pkl
gate() { local NAME=$1 C=$2; for i in $(seq 0 11); do [ -f kd_blocks/${NAME}_b$i.json ] || python3 e12_ens_gate.py --cand $C --ref $A --seeds 500 --workers 100 --seed0 $((360000 + i*1000)) --out kd_blocks/${NAME}_b$i.json; done; }
# quarterens ready now
gate quarterens ckpt/paperx/quarter_s0.pkl,ckpt/paperx/quarter_s1.pkl,ckpt/paperx/quarter_s2.pkl
# wait for retrains then the alpha gates
until [ -f ckpt/paperx/a05_r0.pkl ] && [ -f ckpt/paperx/a10_r0.pkl ] && [ -f ckpt/paperx/a10_r1.pkl ] && [ -f ckpt/paperx/a09_r0.pkl ]; do sleep 120; done
sleep 30
gate a05ens ckpt/paperx/a05_s0.pkl,ckpt/paperx/a05_s1.pkl,ckpt/paperx/a05_r0.pkl
gate a09ens ckpt/paperx/a09_s0.pkl,ckpt/paperx/a09_s3.pkl,ckpt/paperx/a09_r0.pkl
gate a10ens ckpt/paperx/a10_s2.pkl,ckpt/paperx/a10_r0.pkl,ckpt/paperx/a10_r1.pkl
echo FINAL_GATES_DONE
