"""bestnet_val.py CKPT CH BLK TAG — masked top-1 val acc on the SAME seed-12345 50k val split
(comparable to bn128s1 0.887). Writes ckpt/best/val/{TAG}.json."""
import os,sys,json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from models_explore import build
torch.set_num_threads(8)
HERE=os.path.dirname(os.path.abspath(__file__)); DDIR=os.path.join(HERE,"data")
ckpt,ch,blk,tag=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
d=np.load(os.path.join(DDIR,"cooked_single.npz")); o,m,ac=d["obs"],d["mask"],d["act"].astype(np.int64)
N=len(ac); rng=np.random.RandomState(12345); perm=rng.permutation(N)
nval=min(50000,N//20); vidx=np.sort(perm[:nval])
net=build("resbn_fused",channels=ch,blocks=blk); net.load_state_dict(torch.load(ckpt,map_location="cpu")); net.eval()
c=0
with torch.no_grad():
  for i in range(0,len(vidx),8192):
    b=vidx[i:i+8192]
    ob=torch.from_numpy(np.ascontiguousarray(o[b])).float()
    mk=torch.from_numpy(np.ascontiguousarray(m[b])).float()
    y=torch.from_numpy(np.ascontiguousarray(ac[b]))
    pr=net({"is_training":False,"obs":{"observation":ob,"action_mask":mk}}).argmax(1)
    c+=(pr==y).sum().item()
va=c/len(vidx)
os.makedirs(os.path.join(HERE,"ckpt","best","val"),exist_ok=True)
json.dump({"tag":tag,"ckpt":ckpt,"channels":ch,"blocks":blk,"val_acc":round(va,4),"n_val":len(vidx)},
          open(os.path.join(HERE,"ckpt","best","val",tag+".json"),"w"),indent=2)
print(tag,"val_acc",round(va,4))
