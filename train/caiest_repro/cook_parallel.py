"""
cook_parallel.py — parallel version of preprocess_single.py. Splits data.txt at Match
boundaries into chunks, processes each chunk in a worker (EXACT same per-decision extraction
logic), concatenates, writes memmap .npy (cooked_obs/mask/act) + compressed cooked_single.npz.
"""
import os, sys, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
from feature import FeatureAgent

HERE=os.path.dirname(os.path.abspath(__file__))
DDIR=os.path.join(HERE,"data")
DATA=os.path.join(DDIR,"data.txt")

def process_lines(lines):
    all_obs, all_mask, all_act = [], [], []
    obs=[[] for _ in range(4)]; actions=[[] for _ in range(4)]
    agents=None; curTile=None
    def flush():
        for j in range(4):
            for i,a in enumerate(actions[j]):
                o=obs[j][i]
                if np.sum(o["action_mask"])!=1:
                    all_obs.append(o["observation"].astype(np.int8))
                    all_mask.append(o["action_mask"].astype(np.bool_))
                    all_act.append(a)
    for line in lines:
        t=line.split()
        if not t: continue
        if t[0]=="Match":
            agents=[FeatureAgent(i) for i in range(4)]
        elif t[0]=="Wind":
            for ag in agents: ag.request2obs(line.strip())
        elif t[0]=="Player":
            p=int(t[1])
            if t[2]=="Deal":
                agents[p].request2obs(" ".join(t[2:]))
            elif t[2]=="Draw":
                for i in range(4):
                    if i==p:
                        obs[p].append(agents[p].request2obs(" ".join(t[2:]))); actions[p].append(0)
                    else:
                        agents[i].request2obs(" ".join(t[:3]))
            elif t[2]=="Play":
                actions[p].pop(); actions[p].append(agents[p].response2action(" ".join(t[2:])))
                for i in range(4):
                    if i==p: agents[p].request2obs(line.strip())
                    else:
                        obs[i].append(agents[i].request2obs(line.strip())); actions[i].append(0)
                curTile=t[3]
            elif t[2]=="Chi":
                actions[p].pop(); actions[p].append(agents[p].response2action("Chi %s %s"%(curTile,t[3])))
                for i in range(4):
                    if i==p:
                        obs[p].append(agents[p].request2obs("Player %d Chi %s"%(p,t[3]))); actions[p].append(0)
                    else:
                        agents[i].request2obs("Player %d Chi %s"%(p,t[3]))
            elif t[2]=="Peng":
                actions[p].pop(); actions[p].append(agents[p].response2action("Peng %s"%t[3]))
                for i in range(4):
                    if i==p:
                        obs[p].append(agents[p].request2obs("Player %d Peng %s"%(p,t[3]))); actions[p].append(0)
                    else:
                        agents[i].request2obs("Player %d Peng %s"%(p,t[3]))
            elif t[2]=="Gang":
                actions[p].pop(); actions[p].append(agents[p].response2action("Gang %s"%t[3]))
                for i in range(4):
                    agents[i].request2obs("Player %d Gang %s"%(p,t[3]))
            elif t[2]=="AnGang":
                actions[p].pop(); actions[p].append(agents[p].response2action("AnGang %s"%t[3]))
                for i in range(4):
                    if i==p: agents[p].request2obs("Player %d AnGang %s"%(p,t[3]))
                    else: agents[i].request2obs("Player %d AnGang"%p)
            elif t[2]=="BuGang":
                actions[p].pop(); actions[p].append(agents[p].response2action("BuGang %s"%t[3]))
                for i in range(4):
                    if i==p: agents[p].request2obs("Player %d BuGang %s"%(p,t[3]))
                    else:
                        obs[i].append(agents[i].request2obs("Player %d BuGang %s"%(p,t[3]))); actions[i].append(0)
            elif t[2]=="Hu":
                actions[p].pop(); actions[p].append(agents[p].response2action("Hu"))
            if t[2] in ["Peng","Gang","Hu"]:
                for k in range(5,15,5):
                    if len(t)>k:
                        p=int(t[k+1])
                        if t[k+2]=="Chi":
                            actions[p].pop(); actions[p].append(agents[p].response2action("Chi %s %s"%(curTile,t[k+3])))
                        elif t[k+2]=="Peng":
                            actions[p].pop(); actions[p].append(agents[p].response2action("Peng %s"%t[k+3]))
                        elif t[k+2]=="Gang":
                            actions[p].pop(); actions[p].append(agents[p].response2action("Gang %s"%t[k+3]))
                        elif t[k+2]=="Hu":
                            actions[p].pop(); actions[p].append(agents[p].response2action("Hu"))
                    else: break
        elif t[0]=="Score":
            flush()
            for x in obs: x.clear()
            for x in actions: x.clear()
    if not all_obs:
        return (np.zeros((0,38,4,9),np.int8), np.zeros((0,235),np.bool_), np.zeros((0,),np.int16))
    return (np.stack(all_obs).reshape((-1,38,4,9)).astype(np.int8),
            np.stack(all_mask).astype(np.bool_),
            np.array(all_act,dtype=np.int16))

def _worker(chunk):
    try:
        return process_lines(chunk)
    except Exception as e:
        sys.stderr.write("worker err: %s\n"%e)
        return (np.zeros((0,38,4,9),np.int8), np.zeros((0,235),np.bool_), np.zeros((0,),np.int16))

def split_chunks(path, nchunks):
    # read all lines, split into ~equal groups at Match boundaries
    with open(path,encoding="UTF-8") as f:
        lines=f.readlines()
    starts=[i for i,l in enumerate(lines) if l.startswith("Match")]
    nm=len(starts)
    per=max(1,(nm+nchunks-1)//nchunks)
    chunks=[]
    for c in range(0,nm,per):
        a=starts[c]
        b=starts[c+per] if c+per<nm else len(lines)
        chunks.append(lines[a:b])
    return chunks, nm

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--workers",type=int,default=64)
    ap.add_argument("--memmap",action="store_true",help="also write cooked_obs/mask/act .npy memmaps")
    a=ap.parse_args()
    t0=time.time()
    chunks,nm=split_chunks(DATA,a.workers*2)
    print(f"matches {nm} split into {len(chunks)} chunks; workers {a.workers}",flush=True)
    with mp.Pool(a.workers) as p:
        res=p.map(_worker, chunks)
    obs=np.concatenate([r[0] for r in res],axis=0)
    mask=np.concatenate([r[1] for r in res],axis=0)
    act=np.concatenate([r[2] for r in res],axis=0)
    print(f"total samples {len(act):,}  obs{obs.shape} mask{mask.shape} act{act.shape}  ({time.time()-t0:.0f}s)",flush=True)
    h=hashlib.sha256(); h.update(obs.tobytes()); h.update(mask.tobytes()); h.update(act.tobytes())
    print(f"CONTENT_SHA256 {h.hexdigest()}",flush=True)
    np.savez_compressed(os.path.join(DDIR,"cooked_single.npz"),obs=obs,mask=mask,act=act)
    print("wrote cooked_single.npz",flush=True)
    if a.memmap:
        np.save(os.path.join(DDIR,"cooked_obs.npy"),obs)
        np.save(os.path.join(DDIR,"cooked_mask.npy"),mask)
        np.save(os.path.join(DDIR,"cooked_act.npy"),act)
        print("wrote memmap .npy",flush=True)
    print(f"DONE total {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    main()
