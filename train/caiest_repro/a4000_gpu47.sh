#!/bin/bash
# A4000 GPUs 4-7: ladder-field clone + field-adapted kd students (light fine-tune),
# then paired gate of the adapted ensemble.
S="scp -i /root/.ssh/id_main -o StrictHostKeyChecking=no -P 18595"
M=root@175.155.64.222
cd /root/caiest_repro
$S $M:/root/IJCAI-mahjong/data/processed/sim11_cai_danger_bc.npz data/ 2>/dev/null
$S $M:/root/IJCAI-mahjong/data/processed/ladder_cai_danger_bc.npz data/ 2>/dev/null
$S $M:/root/IJCAI-mahjong/train/caiest_repro/e18_finetune.py . 2>/dev/null
$S "$M:/root/IJCAI-mahjong/train/caiest_repro/ckpt/kd/kd_128x40_s0.bn.pkl" ckpt/kd/ 2>/dev/null
$S "$M:/root/IJCAI-mahjong/train/caiest_repro/ckpt/kd/kd_128x40_s1.bn.pkl" ckpt/kd/ 2>/dev/null
$S "$M:/root/IJCAI-mahjong/train/caiest_repro/ckpt/kd/kd_128x40_s2.bn.pkl" ckpt/kd/ 2>/dev/null
$S "$M:/root/IJCAI-mahjong/train/caiest_repro/ckpt/aug/aug_128x40_s0.bn.pkl" ckpt/aug/ 2>/dev/null
mkdir -p ckpt/danger ckpt/kdfield logs
CUDA_VISIBLE_DEVICES=4 nohup python3 e18_finetune.py --channels 128 --blocks 40 --steps 20000 --lr 5e-5 --seed 77 --init ckpt/aug/aug_128x40_s0.bn.pkl --data data/ladder_cai_danger_bc.npz --out ckpt/danger/fieldclone_ladder.pkl > logs/e18_fcladder.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 e18_finetune.py --channels 128 --blocks 40 --steps 4000 --lr 2e-5 --seed 80 --init ckpt/kd/kd_128x40_s0.bn.pkl --data data/sim11_cai_danger_bc.npz --out ckpt/kdfield/kdf_s0.pkl > logs/e18_kdf0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python3 e18_finetune.py --channels 128 --blocks 40 --steps 4000 --lr 2e-5 --seed 81 --init ckpt/kd/kd_128x40_s1.bn.pkl --data data/sim11_cai_danger_bc.npz --out ckpt/kdfield/kdf_s1.pkl > logs/e18_kdf1.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python3 e18_finetune.py --channels 128 --blocks 40 --steps 4000 --lr 2e-5 --seed 82 --init ckpt/kd/kd_128x40_s2.bn.pkl --data data/sim11_cai_danger_bc.npz --out ckpt/kdfield/kdf_s2.pkl > logs/e18_kdf2.log 2>&1 &
until [ -f ckpt/kdfield/kdf_s0.pkl ] && [ -f ckpt/kdfield/kdf_s1.pkl ] && [ -f ckpt/kdfield/kdf_s2.pkl ]; do sleep 180; done
A=ckpt/aug/aug_128x40_s0.pkl
KF=ckpt/kdfield/kdf_s0.pkl,ckpt/kdfield/kdf_s1.pkl,ckpt/kdfield/kdf_s2.pkl
mkdir -p kd_blocks
for i in $(seq 0 11); do
  while [ "$(awk '{print ($1>120)?1:0}' /proc/loadavg)" = "1" ]; do sleep 60; done
  [ -f kd_blocks/kdfield_b$i.json ] || python3 e12_ens_gate.py --cand $KF --ref $A --seeds 500 --workers 100 --seed0 $((360000 + i*1000)) --out kd_blocks/kdfield_b$i.json
done
echo KDFIELD_DONE
