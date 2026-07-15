#!/bin/bash
cd /root/caiest_repro
A=ckpt/aug/aug_128x40_s0.pkl
gate() { local NAME=$1 C=$2; for i in $(seq 0 11); do [ -f kd_blocks/${NAME}_b$i.json ] || python3 e12_ens_gate.py --cand $C --ref $A --seeds 500 --workers 110 --seed0 $((360000 + i*1000)) --out kd_blocks/${NAME}_b$i.json; done; }
gate a05ens ckpt/paperx/a05_s0.pkl,ckpt/paperx/a05_s1.pkl,ckpt/paperx/a05_s2.pkl
gate quarterens ckpt/paperx/quarter_s1.pkl,ckpt/paperx/quarter_s2.pkl,ckpt/paperx/quarter_s3.pkl
gate djAens ckpt/paperx/djA_s0.pkl,ckpt/paperx/djA_s1.pkl,ckpt/paperx/djA_s2.pkl
gate djBens ckpt/paperx/djB_s0.pkl,ckpt/paperx/djB_s1.pkl,ckpt/paperx/djB_s2.pkl
gate djXens ckpt/paperx/djA_s0.pkl,ckpt/paperx/djA_s1.pkl,ckpt/paperx/djB_s0.pkl
echo GATES_NIGHT2_DONE
