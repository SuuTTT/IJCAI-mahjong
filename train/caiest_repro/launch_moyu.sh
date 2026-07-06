#!/bin/bash
cd /root/IJCAI-mahjong/train/caiest_repro
setsid nohup python3 awr_critic.py --gpu 1 --beta 0.5 --steps 10000 --mix 0.3 --lr 5e-5 --out ckpt/rl/moyu_critic_b05.pkl > logs/awr_b05.log 2>&1 < /dev/null &
setsid nohup python3 awr_critic.py --gpu 2 --beta 1.0 --steps 10000 --mix 0.3 --lr 5e-5 --out ckpt/rl/moyu_critic_b1.pkl  > logs/awr_b1.log  2>&1 < /dev/null &
setsid nohup python3 awr_critic.py --gpu 3 --beta 2.0 --steps 10000 --mix 0.3 --lr 5e-5 --out ckpt/rl/moyu_critic_b2.pkl  > logs/awr_b2.log  2>&1 < /dev/null &
sleep 3
echo "launched, running=$(pgrep -fc awr_critic.py)"
