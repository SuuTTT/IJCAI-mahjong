#!/bin/bash
# after kd14 students finish: 3 merged-data field-adapted kd students + 1 merged danger head
S="scp -i /root/.ssh/id_main -o StrictHostKeyChecking=no -P 18595"
M=root@175.155.64.222
cd /root/caiest_repro
until [ -f ckpt/kd14/kd14_s3.pkl ]; do sleep 180; done
$S $M:/root/IJCAI-mahjong/data/processed/all_cai_bc.npz data/
$S $M:/root/IJCAI-mahjong/data/processed/all_cai_danger.npz data/
mkdir -p ckpt/kdfield ckpt/danger
CUDA_VISIBLE_DEVICES=0 nohup python3 e18_finetune.py --channels 128 --blocks 40 --steps 4000 --lr 2e-5 --seed 85 --init ckpt/kd/kd_128x40_s0.bn.pkl --data data/all_cai_bc.npz --out ckpt/kdfield/kdfm_s0.pkl > logs/e18_kdfm0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 e18_finetune.py --channels 128 --blocks 40 --steps 4000 --lr 2e-5 --seed 86 --init ckpt/kd/kd_128x40_s1.bn.pkl --data data/all_cai_bc.npz --out ckpt/kdfield/kdfm_s1.pkl > logs/e18_kdfm1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 e18_finetune.py --channels 128 --blocks 40 --steps 4000 --lr 2e-5 --seed 87 --init ckpt/kd/kd_128x40_s2.bn.pkl --data data/all_cai_bc.npz --out ckpt/kdfield/kdfm_s2.pkl > logs/e18_kdfm2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 PW=30 nohup python3 /root/caiest_repro/e19_danger.py --channels 128 --blocks 40 --steps 20000 --lr 1e-4 --seed 88 --valevery 1000000 --init ckpt/aug/aug_128x40_s0.bn.pkl --data data/all_cai_danger.npz --out ckpt/danger/danger4.pkl > logs/e19_d4.log 2>&1 &
until [ -f ckpt/kdfield/kdfm_s0.pkl ] && [ -f ckpt/kdfield/kdfm_s1.pkl ] && [ -f ckpt/kdfield/kdfm_s2.pkl ]; do sleep 180; done
A=ckpt/aug/aug_128x40_s0.pkl
KFM=ckpt/kdfield/kdfm_s0.pkl,ckpt/kdfield/kdfm_s1.pkl,ckpt/kdfield/kdfm_s2.pkl
for i in $(seq 0 11); do
  while [ "$(awk "{print (\$1>120)?1:0}" /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f kd_blocks/kdfm_b$i.json ] || python3 e12_ens_gate.py --cand $KFM --ref $A --seeds 500 --workers 100 --seed0 $((360000 + i*1000)) --out kd_blocks/kdfm_b$i.json
done
echo KDFM_DONE
