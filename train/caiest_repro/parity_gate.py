"""
parity_gate.py — bias-corrected DUPLICATE strength gate vs moyu (the ONLY number that counts).
Runs 2v2 rotated duplicate games through the fast in-process sim_cnn.Sim with CNN policies.
For each seed it plays BOTH seatings (cand@{0,2} and cand@{1,3}) on the SAME wall -> cancels
seat bias. EDGE/g = candidate_net_total / total_games. Calibration: cand==moyu must give +0.000/g.

Writes a JSON gate file. n=0 games => FAIL (printed, json games=0).

  python3 parity_gate.py --cand CAND.pkl --cand-kind resbn --cand-cfg blocks=40 \
      --ref moyu.pkl --ref-kind resbn --ref-cfg blocks=40 \
      --games 2000 --workers 64 --seed0 50000 --out gate.json
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, multiprocessing as mp
from sim_cnn import Sim
from models_explore import build

torch.set_num_threads(1)
_CACHE = {}

def _parse_cfg(s):
    cfg={}
    for kv in s.split(","):
        kv=kv.strip()
        if not kv: continue
        k,v=kv.split("="); cfg[k]=int(v)
    return cfg

def _load(path, kind, cfg):
    key=(path,kind,tuple(sorted(cfg.items())))
    if key not in _CACHE:
        m=build(kind,**cfg)
        if path.endswith(".npz"):
            z=np.load(path)
            sd={k:torch.from_numpy(z[k]) for k in z.files if z[k].ndim>=1 and k!="meta_blocks"}
            m.load_state_dict(sd, strict=False); m.eval(); return m
        else:
            sd=torch.load(path, map_location="cpu")
            if isinstance(sd,dict) and "state_dict" in sd and not any(k.startswith(("stem","body","foot","head")) for k in sd): sd=sd["state_dict"]
        m.load_state_dict(sd); m.eval()
        _CACHE[key]=m
    return _CACHE[key]

def _greedy(m):
    def fn(obs, mask):
        with torch.no_grad():
            lg=m({"is_training":False,"obs":{
                "observation":torch.from_numpy(np.ascontiguousarray(obs)),
                "action_mask":torch.from_numpy(np.ascontiguousarray(mask))}})
        return [int(lg.numpy().flatten().argmax())]
    return fn

_G={}
def _work(arg):
    seed, cand, ck, ccfg, ref, rk, rcfg = arg
    fc=_greedy(_load(cand,ck,ccfg)); fr=_greedy(_load(ref,rk,rcfg))
    # seating A: cand @ {0,2}
    sim=Sim([fc,fr,fc,fr], seed=seed, quan=0, learner_seats=[0,2], cnn=True); sim.play()
    a_cand=sim.scores[0]+sim.scores[2]; a_ref=sim.scores[1]+sim.scores[3]
    # seating B: cand @ {1,3}, same wall
    sim=Sim([fr,fc,fr,fc], seed=seed, quan=0, learner_seats=[1,3], cnn=True); sim.play()
    b_cand=sim.scores[1]+sim.scores[3]; b_ref=sim.scores[0]+sim.scores[2]
    return a_cand, a_ref, b_cand, b_ref

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cand",required=True); ap.add_argument("--cand-kind",default="resbn"); ap.add_argument("--cand-cfg",default="channels=128,blocks=40")
    ap.add_argument("--ref",required=True);  ap.add_argument("--ref-kind",default="resbn");  ap.add_argument("--ref-cfg",default="channels=128,blocks=40")
    ap.add_argument("--games",type=int,default=2000); ap.add_argument("--workers",type=int,default=64); ap.add_argument("--seed0",type=int,default=50000)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    ccfg=_parse_cfg(a.cand_cfg); rcfg=_parse_cfg(a.ref_cfg)
    args=[(a.seed0+i, a.cand,a.cand_kind,ccfg, a.ref,a.ref_kind,rcfg) for i in range(a.games)]
    t0=time.time()
    with mp.Pool(a.workers) as p:
        res=p.map(_work, args, chunksize=4)
    # each seed contributes 2 games (two seatings)
    cand_net=sum(r[0]+r[2] for r in res); ref_net=sum(r[1]+r[3] for r in res)
    ngames=len(res)*2
    seatA_edge=sum(r[0]-r[1] for r in res)/len(res) if res else 0.0
    seatB_edge=sum(r[2]-r[3] for r in res)/len(res) if res else 0.0
    edge=(cand_net-ref_net)/ngames if ngames else 0.0
    out=dict(cand=os.path.basename(a.cand),ref=os.path.basename(a.ref),
             games=ngames, seeds=len(res), cand_net=int(cand_net), ref_net=int(ref_net),
             edge_per_game=round(edge,4), seatA_edge_per_seed=round(seatA_edge,4),
             seatB_edge_per_seed=round(seatB_edge,4), seconds=round(time.time()-t0,1),
             cand_cfg=a.cand_cfg, ref_cfg=a.ref_cfg, seed0=a.seed0)
    with open(a.out,"w") as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2),flush=True)
    if ngames==0: print("FAIL: n=0 games",flush=True); sys.exit(2)

if __name__=="__main__":
    main()
