import numpy as np, torch, glob, os, json
import models_explore as ME

d=np.load("data/teachers/claim_states.npz")
obs=torch.from_numpy(d["obs"]).float()       # (N,38,4,9)
mask=torch.from_numpy(d["mask"]).bool()       # (N,235)
act=d["act"]
N=len(act)
teacher_claim=float(((act>=36)&(act<133)).mean())

dev="cuda:0" if torch.cuda.is_available() else "cpu"

def chan_from(name):
    if "c256" in name: return 256
    if "c320" in name: return 320
    return 128  # moyu

files=sorted(glob.glob("ckpt/claimsel/cs_*.pkl"))
NEG=-1e9
results={}
mask_dev=mask.to(dev)
inf_mask=torch.where(mask_dev, torch.zeros_like(mask_dev,dtype=torch.float32), torch.full_like(mask_dev,NEG,dtype=torch.float32))
claim_idx=torch.arange(36,133,device=dev)
for f in files:
    name=os.path.basename(f)[:-4]
    C=chan_from(name)
    net=ME.build("resbn_fused",channels=C,blocks=40).eval().to(dev)
    sd=torch.load(f,map_location=dev)
    net.load_state_dict(sd)
    preds=[]
    bs=512
    with torch.no_grad():
        for i in range(0,N,bs):
            x=obs[i:i+bs].to(dev)
            logits=net.foot(net.body(net.stem(x)))   # (B,235)
            logits=logits+inf_mask[i:i+bs]
            am=logits.argmax(1)
            preds.append(am.cpu())
    pred=torch.cat(preds).numpy()
    claim_rate=float(((pred>=36)&(pred<133)).mean())
    chi_rate=float(((pred>=36)&(pred<99)).mean())
    peng_rate=float(((pred>=99)&(pred<133)).mean())
    pass_rate=float((pred==0).mean())
    hu_rate=float((pred==1).mean())
    discard_rate=float(((pred>=2)&(pred<36)).mean())
    gang_rate=float((pred>=133).mean())
    results[name]={"claim_rate":claim_rate,"chi":chi_rate,"peng":peng_rate,
                   "pass":pass_rate,"hu":hu_rate,"discard":discard_rate,"gang":gang_rate}
    print(f"{name:24s} claim={claim_rate:.4f} (chi={chi_rate:.4f} peng={peng_rate:.4f}) pass={pass_rate:.4f} discard={discard_rate:.4f} hu={hu_rate:.4f} gang={gang_rate:.4f}")
    del net; torch.cuda.empty_cache()

print()
print(f"TEACHER (leaders) claim-rate = {teacher_claim:.4f}  TARGET ~0.246")
json.dump({"teacher":teacher_claim,"target":0.246,"results":results}, open("claimrate_results.json","w"), indent=2)
