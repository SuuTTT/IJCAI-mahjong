#!/bin/bash
cd /root/caiest_repro
T1=ckpt/aug/aug_128x40_s0.pkl
T4=ckpt/aug/aug_128x40_s0.pkl,ckpt/aug/aug_128x40_s1.pkl,ckpt/aug/aug_128x40_s2.pkl,ckpt/aug/aug_128x40_s3.pkl
mkdir -p ckpt/kdcurve
CUDA_VISIBLE_DEVICES=4 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 114 --teachers $T4 --alpha 0.7 --out ckpt/kdcurve/kd4t_s1.pkl > logs/kd4t_s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 115 --teachers $T4 --alpha 0.7 --out ckpt/kdcurve/kd4t_s2.pkl > logs/kd4t_s2.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 116 --teachers $T1 --alpha 0.7 --out ckpt/kdcurve/kd1t_s0.pkl > logs/kd1t_s0.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 117 --teachers $T1 --alpha 0.7 --out ckpt/kdcurve/kd1t_s1.pkl > logs/kd1t_s1.log 2>&1 &
wait
echo CURVE_A4000_DONE
