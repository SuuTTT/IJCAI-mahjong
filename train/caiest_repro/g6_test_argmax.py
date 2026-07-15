import os, sys
sys.path.insert(0, "/root/caiest_repro")
os.chdir("/root/caiest_repro")
import numpy as np, torch
torch.set_num_threads(1)
import models_explore
models_explore.IN_PLANES = 38
from models_explore import build
from sim_cnn import Sim, ACT, TILE_LIST

KD = ["ckpt/kd/kd_128x40_s0.pkl", "ckpt/kd/kd_128x40_s1.pkl", "ckpt/kd/kd_128x40_s2.pkl"]
mods=[]
for p in KD:
    m=build("resbn_fused",channels=128,blocks=40); m.load_state_dict(torch.load(p,map_location="cpu")); m.eval(); mods.append(m)
gmods=[build("resbn_fused",channels=128,blocks=40) for _ in range(3)]
for gm,m in zip(gmods,mods): gm.load_state_dict(m.state_dict()); gm.eval(); gm.cuda()

def cpu_ens_logits(obs,mask):
    mk=mask.flatten().astype(np.float32); acc=None
    for m in mods:
        ob=np.ascontiguousarray(obs)[None].astype(np.float32)
        with torch.no_grad():
            lg=m({"is_training":False,"obs":{"observation":torch.from_numpy(ob),"action_mask":torch.from_numpy(np.ascontiguousarray(mask)[None])}}).numpy().flatten()
        lg=np.where(mk>0,lg,-1e30); lg=lg-lg.max(); p=np.exp(lg)*(mk>0); s=p.sum()
        p=p/s if s>0 else (mk/max(1.0,mk.sum())); acc=p if acc is None else acc+p
    avg=acc/len(mods); return np.log(np.where(avg>0,avg,1e-30))

# collect states by playing kdens3 games
def kd_policy(obs,mask):
    lg=cpu_ens_logits(obs[0],mask[0]); a=int(lg.argmax())
    if not mask[0][a]: a=int(np.flatnonzero(mask[0])[0])
    return np.array([a])

states=[]
for seed in range(6):
    sim=Sim([kd_policy]*4, seed=9800000+seed, quan=0, cnn=True)
    sim.reset()
    orig_obs_mask=sim._obs_mask
    def patched(seat, _o=orig_obs_mask):
        obs,mask=_o(seat)
        if mask.sum()>0: states.append((obs.copy(),mask.copy()))
        return obs,mask
    sim._obs_mask=patched
    try: sim._loop(300)
    except Exception as e: pass
print("collected",len(states),"states")

# batched GPU ens logits
def gpu_ens_logits_batch(obs_b, mask_b):
    ob=torch.from_numpy(np.ascontiguousarray(obs_b).astype(np.float32)).cuda()
    mk=torch.from_numpy(np.ascontiguousarray(mask_b)).cuda()
    acc=None
    for gm in gmods:
        with torch.no_grad():
            lg=gm({"is_training":False,"obs":{"observation":ob,"action_mask":mk}})
        mkf=(mk>0).float()
        lg=torch.where(mk>0, lg, torch.tensor(-1e30,device=lg.device))
        lg=lg-lg.max(dim=1,keepdim=True).values
        p=torch.exp(lg)*mkf; s=p.sum(dim=1,keepdim=True)
        p=torch.where(s>0, p/s, mkf/mkf.sum(dim=1,keepdim=True).clamp(min=1.0))
        acc=p if acc is None else acc+p
    avg=acc/len(gmods)
    return torch.log(torch.where(avg>0,avg,torch.tensor(1e-30,device=avg.device)))

obs_b=np.stack([s[0] for s in states]); mask_b=np.stack([s[1] for s in states])
glog=gpu_ens_logits_batch(obs_b,mask_b).cpu().numpy()
def argmax_with_fb(lg,mask):
    a=int(lg.argmax())
    if not mask[a]: a=int(np.flatnonzero(mask)[0])
    return a
mism=0; disc_mism=0; disc_tot=0
for i,(obs,mask) in enumerate(states):
    clg=cpu_ens_logits(obs,mask)
    ca=argmax_with_fb(clg,mask.flatten()); ga=argmax_with_fb(glog[i],mask.flatten())
    is_disc = ACT["Play"]<=ca<ACT["Chi"]
    if is_disc: disc_tot+=1
    if ca!=ga:
        mism+=1
        if is_disc: disc_mism+=1
print(f"total {len(states)} argmax mismatch {mism} ; discard-decisions {disc_tot} discard-mismatch {disc_mism}")
# also max logit diff
print("max abs logit diff", float(np.max(np.abs(glog-np.stack([cpu_ens_logits(o,m) for o,m in states])))))
