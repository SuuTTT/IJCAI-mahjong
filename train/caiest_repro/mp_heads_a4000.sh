#!/bin/bash
cd /root/caiest_repro
until [ -f data/all_mp_danger.npz ]; do sleep 120; done
CUDA_VISIBLE_DEVICES=4 nohup python3 e19_danger.py --channels 128 --blocks 40 --steps 20000 --lr 1e-4 --seed 101 --valevery 1000000 --init ckpt/aug/aug_128x40_s0.bn.pkl --data data/all_mp_danger.npz --out ckpt/danger/dmp_s101.pkl > logs/e19_mp101.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 e19_danger.py --channels 128 --blocks 40 --steps 20000 --lr 1e-4 --seed 102 --valevery 1000000 --init ckpt/aug/aug_128x40_s0.bn.pkl --data data/all_mp_danger.npz --out ckpt/danger/dmp_s102.pkl > logs/e19_mp102.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 PW=10 nohup python3 e19_danger.py --channels 128 --blocks 40 --steps 20000 --lr 1e-4 --seed 103 --valevery 1000000 --init ckpt/aug/aug_128x40_s0.bn.pkl --data data/all_mp_danger.npz --out ckpt/danger/dmp_s103.pkl > logs/e19_mp103.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 PW=100 nohup python3 e19_danger.py --channels 128 --blocks 40 --steps 20000 --lr 1e-4 --seed 104 --valevery 1000000 --init ckpt/aug/aug_128x40_s0.bn.pkl --data data/all_mp_danger.npz --out ckpt/danger/dmp_s104.pkl > logs/e19_mp104.log 2>&1 &
wait
