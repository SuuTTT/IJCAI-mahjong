#!/bin/bash
cd /root/IJCAI-mahjong/train/caiest_repro
until [ -f ckpt/aug/aug_128x40_s6.pkl ] && [ -f ckpt/aug/aug_128x40_s7.pkl ] && [ -f ckpt/aug/aug_128x40_s8.pkl ] && [ -f ckpt/aug/aug_128x40_s9.pkl ]; do sleep 120; done
sleep 30
T10=ckpt/aug/aug_128x40_s0.pkl,ckpt/aug/aug_128x40_s1.pkl,ckpt/aug/aug_128x40_s2.pkl,ckpt/aug/aug_128x40_s3.pkl,ckpt/aug/aug_128x40_s4.pkl,ckpt/aug/aug_128x40_s5.pkl,ckpt/aug/aug_128x40_s6.pkl,ckpt/aug/aug_128x40_s7.pkl,ckpt/aug/aug_128x40_s8.pkl,ckpt/aug/aug_128x40_s9.pkl
mkdir -p ckpt/kd10
CUDA_VISIBLE_DEVICES=0 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 38 --teachers $T10 --alpha 0.7 --out ckpt/kd10/kd10_s8.pkl > logs/kd10_s8.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 39 --teachers $T10 --alpha 0.7 --out ckpt/kd10/kd10_s9.pkl > logs/kd10_s9.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 40 --teachers $T10 --alpha 0.7 --out ckpt/kd10/kd10_s10.pkl > logs/kd10_s10.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 41 --teachers $T10 --alpha 0.7 --out ckpt/kd10/kd10_s11.pkl > logs/kd10_s11.log 2>&1 &
sleep 120
pgrep -fc e13_kd_train > logs/main_students.count
