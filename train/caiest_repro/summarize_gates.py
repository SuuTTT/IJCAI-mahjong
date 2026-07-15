import json, glob, os
GD='/root/IJCAI-mahjong/train/caiest_repro/ckpt/rl/gates'
order=['calib','critic_b05','critic_b1','critic_b2','big256_base','big256_critic']
print(f"{'gate':14s} {'pts':>6s} {'1st%':>6s} {'2nd%':>6s} {'3rd%':>6s} {'4th%':>6s}  vs2.5")
for name in order:
    p=os.path.join(GD,name+'.json')
    if not os.path.exists(p):
        print(f"{name:14s}  (pending)"); continue
    d=json.load(open(p))
    pct=d['dist_pct']; pts=d['placement_pts']
    print(f"{name:14s} {pts:6.3f} {pct[0]:6.1f} {pct[1]:6.1f} {pct[2]:6.1f} {pct[3]:6.1f}  {pts-2.5:+.3f}")
