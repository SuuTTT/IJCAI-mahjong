"""
h2h_fast.py — fast in-process position-controlled head-to-head between two fp16 nets.
2x A + 2x B over identical walls, seats rotated (A@{0,2} then A@{1,3}) to cancel seat bias.
Usage: OPENBLAS_NUM_THREADS=1 python3 eval/h2h_fast.py A.npz B.npz [N] [WORKERS]
"""
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, multiprocessing as mp
from train.sim import Sim
from train.numpy_infer import NumpyMLP

_C = {}
def _pol(path):
    if path not in _C:
        _C[path] = NumpyMLP(path)
    m = _C[path]
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

def _g(arg):
    seed, pa, pb, a02 = arg
    fa, fb = _pol(pa), _pol(pb)
    pols = [fa, fb, fa, fb] if a02 else [fb, fa, fb, fa]
    aseats = (0, 2) if a02 else (1, 3)
    sim = Sim(pols, seed=seed, quan=0); _, sc = sim.play()
    a = sum(sc[s] for s in aseats); b = sum(sc[s] for s in range(4) if s not in aseats)
    w = max(range(4), key=lambda i: sc[i]) if max(sc) > 0 else -1
    aw = 1 if w in aseats else 0; bw = 1 if (w != -1 and w not in aseats) else 0
    dr = 1 if w == -1 else 0
    return a, b, aw, bw, dr

if __name__ == "__main__":
    A, B = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    W = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    args = [(s, A, B, (s % 2 == 0)) for s in range(N)]   # each wall played both seatings via parity
    with mp.Pool(W) as p:
        res = p.map(_g, args, chunksize=8)
    na = sum(r[0] for r in res); nb = sum(r[1] for r in res)
    wa = sum(r[2] for r in res); wb = sum(r[3] for r in res); dr = sum(r[4] for r in res)
    print(f"{N} games  A={os.path.basename(A)}  B={os.path.basename(B)}")
    print(f"  A net={na:+d} wins={wa}   B net={nb:+d} wins={wb}   draws={dr} ({100*dr/N:.0f}%)")
    print(f"  => {'A' if na>nb else 'B' if nb>na else 'TIE'} stronger, margin {na-nb:+d}")
