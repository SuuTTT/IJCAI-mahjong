#!/bin/bash
# KD-at-capacity: does KD unlock capacity plain training could not?
# GPU0/1/3 = kd192 s0/s1/s2, GPU2 = kd256 s0. Same 6 aug teachers, alpha 0.7.
cd /root/IJCAI-mahjong/train/caiest_repro
T=ckpt/aug/aug_128x40_s0.pkl,ckpt/aug/aug_128x40_s1.pkl,ckpt/aug/aug_128x40_s2.pkl,ckpt/aug/aug_128x40_s3.pkl,ckpt/aug/aug_128x40_s4.pkl,ckpt/aug/aug_128x40_s5.pkl
mkdir -p ckpt/kdcap
CUDA_VISIBLE_DEVICES=0 nohup python3 e13_kd_train.py --channels 192 --blocks 40 --steps 90000 --seed 0 --teachers $T --alpha 0.7 --out ckpt/kdcap/kd_192x40_s0.pkl > logs/e13_kd192_s0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 e13_kd_train.py --channels 192 --blocks 40 --steps 90000 --seed 1 --teachers $T --alpha 0.7 --out ckpt/kdcap/kd_192x40_s1.pkl > logs/e13_kd192_s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 e13_kd_train.py --channels 192 --blocks 40 --steps 90000 --seed 2 --teachers $T --alpha 0.7 --out ckpt/kdcap/kd_192x40_s2.pkl > logs/e13_kd192_s2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 e13_kd_train.py --channels 256 --blocks 40 --steps 90000 --seed 0 --teachers $T --alpha 0.7 --out ckpt/kdcap/kd_256x40_s0.pkl > logs/e13_kd256_s0.log 2>&1 &
echo "launched: $(jobs -p | wc -l) trainers"
