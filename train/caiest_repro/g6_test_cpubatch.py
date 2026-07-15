import os,sys
sys.path.insert(0,"/root/caiest_repro"); os.chdir("/root/caiest_repro")
import numpy as np, torch
torch.set_num_threads(1)
import models_explore; models_explore.IN_PLANES=38
from models_explore import build
KD=["ckpt/kd/kd_128x40_s0.pkl","ckpt/kd/kd_128x40_s1.pkl","ckpt/kd/kd_128x40_s2.pkl"]
m=build("resbn_fused",channels=128,blocks=40); m.load_state_dict(torch.load(KD[0],map_location="cpu")); m.eval()
rng=np.random.RandomState(0)
# random plausible obs (0/1) and masks
N=64
obs=(rng.rand(N,38,4,9)<0.15).astype(np.float32)
mask=np.zeros((N,235),dtype=bool)
for i in range(N):
    k=rng.randint(1,10); idx=rng.choice(235,k,replace=False); mask[i,idx]=True
# single
single=np.zeros((N,235),dtype=np.float32)
for i in range(N):
    with torch.no_grad():
        single[i]=m({"is_training":False,"obs":{"observation":torch.from_numpy(obs[i:i+1]),"action_mask":torch.from_numpy(mask[i:i+1])}}).numpy()[0]
# batched
with torch.no_grad():
    batched=m({"is_training":False,"obs":{"observation":torch.from_numpy(obs),"action_mask":torch.from_numpy(mask)}}).numpy()
d=np.abs(single-batched)
d=d[np.isfinite(d)]
print("CPU single-vs-batched max abs diff:", float(d.max()), "num exact equal fraction:", float((d==0).mean()))
# also value net
from f2_value_v2 import VNet
v=VNet(cond=True); 
import glob
try:
    v.load_state_dict(torch.load("results/VALUE_C_60K.pt",map_location="cpu")); v.eval()
    src=torch.tensor([0]*N,dtype=torch.long)
    xs=torch.from_numpy(obs)
    vs_single=np.array([float(v(xs[i:i+1],src[i:i+1]).item()) for i in range(N)])
    with torch.no_grad(): vs_batch=v(xs,src).numpy()
    dv=np.abs(vs_single-vs_batch)
    print("VNet single-vs-batched max abs diff:", float(dv.max()))
except Exception as e:
    print("vnet err",e)
