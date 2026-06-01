"""
tournament.py — fast in-process round-robin between candidate policies.

Uses the SAME Sim + NumpyMLP forward as deployment (greedy = how the bot plays
in eval), so no subprocess/judge flakiness. For every unordered pair (A,B) we
play 2v2 over a fixed set of walls, with seats rotated (A at {0,2} then {1,3})
to cancel positional bias. Net score = (A's points) - (B's points), summed.

Run:  OPENBLAS_NUM_THREADS=1 python3 eval/tournament.py N_GAMES WORKERS
(models listed in CANDIDATES below; net>0 means row beats column.)
"""
import os, sys, itertools
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import multiprocessing as mp
from train.sim import Sim
from train.numpy_infer import NumpyMLP

CK = "train/checkpoints"
# Core candidates + any fleet-experiment outputs that exist (filtered below).
_ALL = {
    "bc_v3_ft":  f"{CK}/bc_v3_ft_fp16.npz",
    "ppo_vb":    f"{CK}/ppo_vb_fp16.npz",
    "league":    f"{CK}/league_best_weights.npz",
    "drawpush":  f"{CK}/drawpush_best_weights.npz",
    "robust":    f"{CK}/robust_best_weights.npz",
    "poolbig":   f"{CK}/poolbig_best_weights.npz",
    "explore":   f"{CK}/explore_best_weights.npz",
    "bigbatch":  f"{CK}/bigbatch_best_weights.npz",
}
CANDIDATES = {k: v for k, v in _ALL.items() if os.path.exists(v)}

_CACHE = {}
def _pol(path):
    if path not in _CACHE:
        _CACHE[path] = NumpyMLP(path)
    m = _CACHE[path]
    def fn(obs, mask):
        out = []
        for o, mk in zip(obs, mask):
            probs, _ = m.forward(o, mk)
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = np.where(mk, probs, 0.0)
            legal = np.flatnonzero(mk)
            out.append(int(np.argmax(probs)) if probs.sum() > 0
                       else (int(legal[0]) if len(legal) else 0))
        return np.array(out)
    return fn

def _one_game(arg):
    """arg = (seed, pathA, pathB, a_at_02). Returns (a_pts, b_pts, a_win, b_win, draw)."""
    seed, pa, pb, a02 = arg
    fa, fb = _pol(pa), _pol(pb)
    if a02:   # A at seats 0,2 ; B at 1,3
        pols = [fa, fb, fa, fb]; aseats = (0, 2)
    else:
        pols = [fb, fa, fb, fa]; aseats = (1, 3)
    sim = Sim(pols, seed=seed, quan=0)
    _, sc = sim.play()
    a_pts = sum(sc[s] for s in aseats)
    b_pts = sum(sc[s] for s in range(4) if s not in aseats)
    w = max(range(4), key=lambda i: sc[i]) if max(sc) > 0 else -1
    a_win = int(w in aseats); b_win = int(w != -1 and w not in aseats)
    return (a_pts, b_pts, a_win, b_win, int(w == -1))

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 26
    names = list(CANDIDATES)
    print(f"Tournament: {names}  N={N}/pair  workers={W}", flush=True)
    net = {a: {b: 0 for b in names} for a in names}
    wins = {a: 0 for a in names}; total_games = 0
    pool = mp.Pool(W)
    for a, b in itertools.combinations(names, 2):
        pa, pb = CANDIDATES[a], CANDIDATES[b]
        args = [(50000 + g, pa, pb, (g % 2 == 0)) for g in range(N)]
        res = pool.map(_one_game, args, chunksize=4)
        ap = sum(r[0] for r in res); bp = sum(r[1] for r in res)
        aw = sum(r[2] for r in res); bw = sum(r[3] for r in res)
        dr = sum(r[4] for r in res)
        net[a][b] = ap - bp; net[b][a] = bp - ap
        wins[a] += aw; wins[b] += bw; total_games += N
        print(f"  {a:9s} vs {b:9s}: net {ap-bp:+6d}  (wins {aw} vs {bw}, draws {dr}/{N})", flush=True)
    pool.close(); pool.join()
    print("\n=== net score matrix (row - col, + = row stronger) ===")
    print("           " + "".join(f"{b:>10s}" for b in names))
    for a in names:
        print(f"{a:>10s} " + "".join(f"{net[a][b]:>10d}" for b in names))
    print("\n=== aggregate (sum of net vs all others) ===")
    agg = {a: sum(net[a][b] for b in names) for a in names}
    for a in sorted(names, key=lambda x: -agg[x]):
        print(f"  {a:9s}: total net {agg[a]:+7d}   raw wins {wins[a]}")

if __name__ == "__main__":
    main()
