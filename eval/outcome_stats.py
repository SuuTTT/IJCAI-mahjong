"""
outcome_stats.py — classify game outcomes for a policy from score vectors alone.
Rong: winner=+24+f, discarder=-(8+f) (most negative), others=-8.
Self-draw: winner=+24+3f, others=-(8+f).  Draw: all 0.
Reports, for the TARGET policy's seats: win-selfdraw / win-rong / deal-in / bystander-loss / draw.
This tells us how often deal-in even happens -> whether DEFENSE is a measurable lever.

Usage: OPENBLAS_NUM_THREADS=1 python3 eval/outcome_stats.py TARGET.npz OPP.npz [N] [W]
       (target seated 0,2 and 1,3 alternately; opp fills the rest)
"""
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, multiprocessing as mp
from train.sim import Sim
from train.numpy_infer import NumpyMLP

_C = {}
def _pol(path):
    if path not in _C: _C[path] = NumpyMLP(path)
    m = _C[path]
    def fn(obs, mask):
        out = []
        for o, mk in zip(obs, mask):
            probs, _ = m.forward(o, mk)
            probs = np.nan_to_num(probs); probs = np.where(mk, probs, 0.0)
            legal = np.flatnonzero(mk)
            out.append(int(np.argmax(probs)) if probs.sum() > 0 else (int(legal[0]) if len(legal) else 0))
        return np.array(out)
    return fn

def classify(sc):
    """Return (kind, winner, discarder). kind in draw/selfdraw/rong."""
    if max(sc) <= 0: return ("draw", -1, -1)
    w = int(np.argmax(sc))
    negs = [i for i in range(4) if sc[i] < 0]
    if len(set(sc[i] for i in negs)) == 1:   # all losers equal -> self-draw
        return ("selfdraw", w, -1)
    disc = int(np.argmin(sc))                # most negative = discarder
    return ("rong", w, disc)

def _g(arg):
    seed, pt, po, t02 = arg
    ft, fo = _pol(pt), _pol(po)
    pols = [ft, fo, ft, fo] if t02 else [fo, ft, fo, ft]
    tseats = {0, 2} if t02 else {1, 3}
    sim = Sim(pols, seed=seed, quan=0); _, sc = sim.play()
    kind, w, disc = classify(sc)
    r = dict(draw=0, t_selfdraw=0, t_rong=0, t_dealin=0, t_bystd=0, net=0)
    r["net"] = sum(sc[s] for s in tseats)
    if kind == "draw": r["draw"] = 1
    elif kind == "selfdraw":
        if w in tseats: r["t_selfdraw"] = 1
        else: r["t_bystd"] = 1
    else:  # rong
        if w in tseats: r["t_rong"] = 1
        elif disc in tseats: r["t_dealin"] = 1
        else: r["t_bystd"] = 1
    return r

if __name__ == "__main__":
    T = sys.argv[1]; O = sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    W = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    args = [(s, T, O, (s % 2 == 0)) for s in range(N)]
    with mp.Pool(W) as p: res = p.map(_g, args, chunksize=8)
    agg = {k: sum(r[k] for r in res) for r in [res[0]] for k in res[0]}
    for k in res[0]: agg[k] = sum(r[k] for r in res)
    print(f"{N} games  TARGET={os.path.basename(T)}  OPP={os.path.basename(O)}")
    print(f"  draws        {agg['draw']:5d}  ({100*agg['draw']/N:.1f}%)")
    print(f"  T self-draw  {agg['t_selfdraw']:5d}  ({100*agg['t_selfdraw']/N:.1f}%)")
    print(f"  T rong-win   {agg['t_rong']:5d}  ({100*agg['t_rong']/N:.1f}%)")
    print(f"  T DEAL-IN    {agg['t_dealin']:5d}  ({100*agg['t_dealin']/N:.1f}%)   <- defense target")
    print(f"  T bystander  {agg['t_bystd']:5d}  ({100*agg['t_bystd']/N:.1f}%)")
    print(f"  T net (2 seats) {agg['net']:+d}")
