"""
e12_ens_gate.py — SEED-ENSEMBLE gate: mixture of fused nets (deploy rule from
realfield_build/ensemble_infer.py: arithmetic mean of softmax over LEGAL set, then argmax)
vs single reference, in the calibrated duplicate placement gate.
Calibration: --cand with ONE path == --ref must give exactly 2.500 (argmax of log-softmax == raw argmax).
  python3 e12_ens_gate.py --cand a.pkl,b.pkl,c.pkl --ref a.pkl --seeds 500 --workers 48 --seed0 200000 --out X.json
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, multiprocessing as mp
from sim_cnn import Sim
from models_explore import build

torch.set_num_threads(1)
_CACHE = {}

def _load(path, kind="resbn_fused", cfg=None):
    cfg = cfg or {"channels":128,"blocks":40}
    key = (path, kind, tuple(sorted(cfg.items())))
    if key not in _CACHE:
        m = build(kind, **cfg)
        sd = torch.load(path, map_location="cpu")
        m.load_state_dict(sd); m.eval()
        _CACHE[key] = m
    return _CACHE[key]

def _logits(m, obs, mask):
    with torch.no_grad():
        lg = m({"is_training": False, "obs": {
            "observation": torch.from_numpy(np.ascontiguousarray(obs)),
            "action_mask": torch.from_numpy(np.ascontiguousarray(mask))}})
    return lg.numpy().flatten()

def _single_fn(m):
    def fn(obs, mask, return_logits=False):
        lg = _logits(m, obs, mask)
        return lg if return_logits else [int(lg.argmax())]
    return fn

def _ens_fn(models):
    """EXACT deploy rule (ensemble_infer.Ensemble.logits): mean softmax over legal, log."""
    def fn(obs, mask, return_logits=False):
        mk = mask.flatten().astype(np.float32)
        acc = None
        for m in models:
            lg = _logits(m, obs, mask)
            lg = np.where(mk > 0, lg, -1e30)
            lg = lg - lg.max()
            p = np.exp(lg) * (mk > 0)
            s = p.sum()
            p = p / s if s > 0 else (mk / max(1.0, mk.sum()))
            acc = p if acc is None else acc + p
        avg = acc / len(models)
        out = np.log(np.where(avg > 0, avg, 1e-30))
        return out if return_logits else [int(out.argmax())]
    return fn

class PSim(Sim):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
    def _ask(self, seat):
        from sim_cnn import ACT
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        lg = self.policies[seat](obs[None, :], mask[None, :], return_logits=True)
        act = int(lg.argmax())
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        return act

def _parse_spec(spec):
    if "@" in spec:
        p, ch = spec.split("@"); return p, {"channels": int(ch), "blocks": 40}
    return spec, {"channels": 128, "blocks": 40}

def _work(arg):
    seed, cands, ref = arg
    fc = _ens_fn([_load(_parse_spec(p)[0], "resbn_fused", _parse_spec(p)[1]) for p in cands])
    fr = _single_fn(_load(ref))
    placement_sum = 0.0
    for cs in range(4):
        pols = [fr]*4; pols[cs] = fc
        sim = PSim(pols, seed=seed, quan=0, learner_seats=[], cnn=True)
        sim.play()
        sc = sim.scores; c = sc[cs]
        greater = sum(1 for j in range(4) if sc[j] > c)
        equal = sum(1 for j in range(4) if sc[j] == c)
        placement_sum += 5.0 - (greater + (equal + 1) / 2.0)
    return placement_sum

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)   # comma-separated fused pkls
    ap.add_argument("--ref", required=True)
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=200000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cands = a.cand.split(",")
    args = [(a.seed0 + i, cands, a.ref) for i in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers) as p:
        res = p.map(_work, args, chunksize=1)
    ngames = len(res) * 4
    pts = sum(res) / ngames if ngames else 0.0
    out = dict(cand=[os.path.basename(c) for c in cands], ref=os.path.basename(a.ref),
               rule="mean-softmax-over-legal (deploy ensemble_infer)", games=ngames,
               placement_pts=round(pts, 4), seconds=round(time.time()-t0, 1), seed0=a.seed0)
    with open(a.out, "w") as f: json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)
    if ngames == 0: sys.exit(2)

if __name__ == "__main__":
    main()
