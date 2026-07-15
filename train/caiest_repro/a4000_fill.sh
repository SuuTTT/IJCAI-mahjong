#!/bin/bash
# GPUs 5-7 now: 3 more merged-data danger heads (committee for ensemble tie-break).
# GPUs 0-3 when kd14 students finish: danger4 + 3 more seeds (committee to 10).
cd /root/caiest_repro
mkdir -p ckpt/danger logs
CUDA_VISIBLE_DEVICES=5 nohup python3 e19_danger.py --channels 128 --blocks 40 --steps 20000 --lr 1e-4 --seed 89 --valevery 1000000 --init ckpt/aug/aug_128x40_s0.bn.pkl --data data/all_cai_danger.npz --out ckpt/danger/dgc_s89.pkl > logs/e19_s89.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python3 e19_danger.py --channels 128 --blocks 40 --steps 20000 --lr 1e-4 --seed 90 --valevery 1000000 --init ckpt/aug/aug_128x40_s0.bn.pkl --data data/all_cai_danger.npz --out ckpt/danger/dgc_s90.pkl > logs/e19_s90.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 PW=100 nohup python3 e19_danger.py --channels 128 --blocks 40 --steps 20000 --lr 1e-4 --seed 91 --valevery 1000000 --init ckpt/aug/aug_128x40_s0.bn.pkl --data data/all_cai_danger.npz --out ckpt/danger/dgc_s91.pkl > logs/e19_s91.log 2>&1 &
until [ -f ckpt/kd14/kd14_s3.pkl ]; do sleep 180; done
CUDA_VISIBLE_DEVICES=0 nohup python3 e19_danger.py --channels 128 --blocks 40 --steps 20000 --lr 1e-4 --seed 88 --valevery 1000000 --init ckpt/aug/aug_128x40_s0.bn.pkl --data data/all_cai_danger.npz --out ckpt/danger/danger4.pkl > logs/e19_d4.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 e19_danger.py --channels 128 --blocks 40 --steps 20000 --lr 1e-4 --seed 92 --valevery 1000000 --init ckpt/aug/aug_128x40_s0.bn.pkl --data data/all_cai_danger.npz --out ckpt/danger/dgc_s92.pkl > logs/e19_s92.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 PW=10 nohup python3 e19_danger.py --channels 128 --blocks 40 --steps 20000 --lr 1e-4 --seed 93 --valevery 1000000 --init ckpt/aug/aug_128x40_s0.bn.pkl --data data/all_cai_danger.npz --out ckpt/danger/dgc_s93.pkl > logs/e19_s93.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 e18_finetune.py --channels 128 --blocks 40 --steps 40000 --lr 5e-5 --seed 94 --init ckpt/aug/aug_128x40_s0.bn.pkl --data data/all_cai_bc.npz --out ckpt/danger/fieldclone_strong2.pkl > logs/e18_fcs2.log 2>&1 &
wait
echo A4000_FILL_DONE
