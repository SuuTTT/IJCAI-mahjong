"""
e12_score_gate.py — DUAL-METRIC duplicate gate: placement points AND raw duplicate score.

Fork of e12_ens_gate.py with two changes:
  1. --ref may be a comma-separated LIST of fused pkls -> deploy ensemble rule
     (mean softmax over legal), same as --cand. Single path -> raw-logits argmax
     (bit-identical to e12_ens_gate --ref).
  2. Reports the Botzone-Final Stage-2 metric alongside placement: mean RAW duplicate
     score per game for cand seats and for ref seats (zimo pays 3x, so score and
     placement can diverge).

Placement calc is IDENTICAL to e12_ens_gate (same PSim, same rotations, same seeds), so
placement numbers are directly comparable to historical e12 gates.
Calibration: --cand == --ref must give placement exactly 2.500 and score_diff exactly 0.

  python3 e12_score_gate.py --cand a.pkl,b.pkl,c.pkl --ref x.pkl[,y.pkl,...] \
      --seeds 500 --workers 48 --seed0 500000 --out X.json
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, multiprocessing as mp
from sim_cnn import Sim
from models_explore import build

torch.set_num_threads(1)
_CACHE = {}


def _load(path, kind="resbn_fused", cfg=None):
    cfg = cfg or {"channels": 128, "blocks": 40}
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


def _policy(paths):
    models = []
    for p in paths:
        pth, cfg = _parse_spec(p)
        models.append(_load(pth, "resbn_fused", cfg))
    return _ens_fn(models) if len(models) > 1 else _single_fn(models[0])


def _work(arg):
    seed, cands, refs = arg
    fc = _policy(cands)
    fr = _policy(refs)
    placement_sum = 0.0
    cand_sc = 0.0        # raw score of cand seat, summed over 4 rotations
    ref_sc = 0.0         # raw score of ref seats, summed over 4 rotations x 3 seats
    for cs in range(4):
        pols = [fr] * 4; pols[cs] = fc
        sim = PSim(pols, seed=seed, quan=0, learner_seats=[], cnn=True)
        sim.play()
        sc = sim.scores; c = sc[cs]
        greater = sum(1 for j in range(4) if sc[j] > c)
        equal = sum(1 for j in range(4) if sc[j] == c)
        placement_sum += 5.0 - (greater + (equal + 1) / 2.0)
        cand_sc += c
        ref_sc += sum(sc[j] for j in range(4) if j != cs)
    return placement_sum, cand_sc, ref_sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)   # comma-separated fused pkls
    ap.add_argument("--ref", required=True)    # comma-separated fused pkls (1 = single)
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=200000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cands = a.cand.split(","); refs = a.ref.split(",")
    args = [(a.seed0 + i, cands, refs) for i in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers) as p:
        res = p.map(_work, args, chunksize=1)
    ngames = len(res) * 4
    pts = sum(r[0] for r in res) / ngames if ngames else 0.0
    csc = sum(r[1] for r in res) / ngames if ngames else 0.0
    rsc = sum(r[2] for r in res) / (3 * ngames) if ngames else 0.0
    secs = time.time() - t0
    out = dict(cand=[os.path.basename(c) for c in cands],
               ref=[os.path.basename(r) for r in refs],
               rule="mean-softmax-over-legal (deploy ensemble_infer); ref same rule if >1",
               games=ngames,
               placement_pts=round(pts, 4),
               cand_score_mean=round(csc, 4),
               ref_score_mean=round(rsc, 4),
               score_diff=round(csc - rsc, 4),
               games_per_hour=round(ngames / (secs / 3600.0), 1),
               seconds=round(secs, 1), seed0=a.seed0)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)
    if ngames == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
