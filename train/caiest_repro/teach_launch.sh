#!/bin/bash
cd /root/IJCAI-mahjong/train/caiest_repro
CUDA_VISIBLE_DEVICES=0 nohup python3 e11_train.py --channels 128 --blocks 40 --steps 90000 --seed 6 --out ckpt/aug/aug_128x40_s6.pkl > logs/e11_s6.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 e11_train.py --channels 128 --blocks 40 --steps 90000 --seed 7 --out ckpt/aug/aug_128x40_s7.pkl > logs/e11_s7.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 e11_train.py --channels 128 --blocks 40 --steps 90000 --seed 8 --out ckpt/aug/aug_128x40_s8.pkl > logs/e11_s8.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 e11_train.py --channels 128 --blocks 40 --steps 90000 --seed 9 --out ckpt/aug/aug_128x40_s9.pkl > logs/e11_s9.log 2>&1 &
sleep 180
pgrep -fc "e11_train" > logs/teach_launch.count
