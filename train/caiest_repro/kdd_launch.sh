#!/bin/bash
cd /root/IJCAI-mahjong/train/caiest_repro
T6=ckpt/aug/aug_128x40_s0.pkl,ckpt/aug/aug_128x40_s1.pkl,ckpt/aug/aug_128x40_s2.pkl,ckpt/aug/aug_128x40_s3.pkl,ckpt/aug/aug_128x40_s4.pkl,ckpt/aug/aug_128x40_s5.pkl
pgrep -f "kdd_lightaug" > /dev/null || CUDA_VISIBLE_DEVICES=1 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 7 --teachers $T6 --alpha 0.7 --p_suit 0.6 --p_ref 0.3 --p_drag 0.3 --out ckpt/kddiv/kdd_lightaug.pkl > logs/e13_kdd_lightaug.log 2>&1 &
pgrep -f "kdd_purekd" > /dev/null || CUDA_VISIBLE_DEVICES=2 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 8 --teachers $T6 --alpha 0.9 --out ckpt/kddiv/kdd_purekd.pkl > logs/e13_kdd_purekd.log 2>&1 &
pgrep -f "kdd_halfkd" > /dev/null || CUDA_VISIBLE_DEVICES=3 nohup python3 e13_kd_train.py --channels 128 --blocks 40 --steps 90000 --seed 9 --teachers $T6 --alpha 0.5 --lsm 0.1 --out ckpt/kddiv/kdd_halfkd.pkl > logs/e13_kdd_halfkd.log 2>&1 &
wait
