"""e4_claimrate.py — claim-rate (chi/peng/gang) of base moyu + each E4 AWR output on claim_states.npz.
Measures whether AWR shifts the policy toward/away from claiming. Writes ckpt/e4/claimrate.json."""
import numpy as np, torch, glob, os, json, sys
sys.path.insert(0,"/root/IJCAI-mahjong/train/caiest_repro")
import models_explore as ME

d=np.load("data/teachers/claim_states.npz")
obs=torch.from_numpy(d["obs"]).float(); mask=torch.from_numpy(d["mask"]).bool()
act=d["act"]; N=len(act)
teacher_claim=float(((act>=36)&(act<133)).mean())
dev="cuda:0" if torch.cuda.is_available() else "cpu"
obs=obs.to(dev); mask=mask.to(dev)
inf=torch.where(mask, torch.zeros_like(mask,dtype=torch.float32), torch.full_like(mask,-1e9,dtype=torch.float32))

def rates(pred):
    return dict(claim=float(((pred>=36)&(pred<133)).mean()),
                chi=float(((pred>=36)&(pred<99)).mean()),
                peng=float(((pred>=99)&(pred<133)).mean()),
                gang=float((pred>=133).mean()),
                hu=float((pred==1).mean()),
                pass_=float((pred==0).mean()),
                discard=float(((pred>=2)&(pred<36)).mean()))

def infer(net, kind):
    preds=[]
    with torch.no_grad():
        for i in range(0,N,1024):
            ob=obs[i:i+1024]; mk=mask[i:i+1024]
            if kind=="resbn":
                lg=net({"is_training":False,"obs":{"observation":ob,"action_mask":mk}})
            else:  # resbn_fused
                lg=net.foot(net.body(net.stem(ob)))+inf[i:i+1024]
            preds.append(lg.argmax(1).cpu())
    return torch.cat(preds).numpy()

results={}
# base moyu (resbn, unfused)
base=ME.build("resbn",channels=128,blocks=40).eval().to(dev)
sd=torch.load("/root/assets/moyu_bn_128x40.pkl",map_location=dev)
if isinstance(sd,dict) and "state_dict" in sd: sd=sd["state_dict"]
base.load_state_dict(sd)
results["base_moyu"]=rates(infer(base,"resbn"))
print("base_moyu", results["base_moyu"]["claim"], flush=True)
del base; torch.cuda.empty_cache()

# E4 outputs (resbn_fused)
for f in sorted(glob.glob("ckpt/e4/awr_b*.pkl")):
    name=os.path.basename(f)[:-4]
    net=ME.build("resbn_fused",channels=128,blocks=40).eval().to(dev)
    net.load_state_dict(torch.load(f,map_location=dev))
    results[name]=rates(infer(net,"resbn_fused"))
    print(name, "claim=%.4f"%results[name]["claim"], flush=True)
    del net; torch.cuda.empty_cache()

json.dump({"teacher_claim":teacher_claim,"N":N,"results":results},
          open("ckpt/e4/claimrate.json","w"), indent=2)
print("wrote ckpt/e4/claimrate.json", flush=True)
