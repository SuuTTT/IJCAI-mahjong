"""
agree_eval.py — fair cross-model ranking proxy that does NOT need the (deadlocking) judge bench.
Scores each candidate's argmax-agreement on a FIXED held-out slice of a teacher npz. Use alltop30's
val split (RandomState(0), same split distill_kl reserves): no candidate trained on it (chun models
never saw alltop30; alltop30 models used it only as val) -> clean, comparable across all candidates.

NOTE (project recipe law): agreement is a PROXY, play gates. s2800 had best agreement yet LOST play.
Treat this as a coarse tiebreaker / sanity gate, NOT the final verdict — the Botzone ladder / A4000
gauntlet is the real yardstick. Higher agreement-with-held-out-strong-decisions is a weak "plays-like
-strong-players" signal; do not over-trust a small gap.

  python3 agree_eval.py --holdout data/alltop30.npz --ckpts ckpt/distill100b_fused.pkl ckpt/ens_*.pkl
"""
import os, sys, glob, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from models_explore import build

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holdout', required=True)
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--kind', default='resbn_fused'); ap.add_argument('--cfg', default='{"channels":128,"blocks":40}')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = json.loads(a.cfg)
    z = np.load(a.holdout); obs, mask, act = z['obs'], z['mask'], z['act'].astype(np.int64)
    # reproduce distill_kl's FIXED val split exactly (RandomState(0), nval=max(50,len//5))
    rng = np.random.RandomState(0); idx = rng.permutation(len(act)); nval = max(50, len(idx)//5)
    vi = idx[:nval]
    vO = torch.from_numpy(obs[vi]).float().to(dev)
    vM = torch.from_numpy(mask[vi]).float().to(dev)
    vA = act[vi]
    print(f"holdout {os.path.basename(a.holdout)}: {len(act)} total -> {nval} val decisions", flush=True)
    paths = []
    for c in a.ckpts: paths += sorted(glob.glob(c))
    net = build(a.kind, **cfg).to(dev); net.eval()
    rows = []
    for p in paths:
        try:
            net.load_state_dict(torch.load(p, map_location=dev))
        except Exception as e:
            print(f"  SKIP {os.path.basename(p)}: {e}"); continue
        with torch.no_grad():
            pr = net({'is_training':False,'obs':{'observation':vO,'action_mask':vM}}).argmax(1).cpu().numpy()
        ag = float((pr == vA).mean())
        rows.append((ag, os.path.basename(p)))
    rows.sort(reverse=True)
    print("\n=== agreement on held-out top-30 decisions (proxy; ladder gates) ===")
    for ag, name in rows: print(f"  {ag:.4f}  {name}")

if __name__ == '__main__':
    main()
