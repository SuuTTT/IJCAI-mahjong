"""
winprob_calibrate.py — validate the heuristic win-prob estimator's CALIBRATION against GROUND TRUTH.

Plays raw self-play games (raw base_resbn40_v3 in all 4 seats), and at EVERY discard turn of EVERY
seat records (win_prob_estimate, seat). At game end, labels each record with did-that-seat-win
(final score > 0). Reports AUC (rank separation of winners vs losers) and Brier (calibration), plus a
reliability table. n=0 records => FAIL.

  python3 winprob_calibrate.py --cand base_resbn40_v3.pkl --games 800 --workers 48 --out calib.json
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, multiprocessing as mp
from sim_cnn import Sim, ACT
from models_explore import build
import winprob_estimator as WP
from feature import FeatureAgent as CaiAgent

torch.set_num_threads(1)
_CACHE = {}
PLAY = CaiAgent.OFFSET_ACT['Play']


def _load(path, kind, cfg):
    key = (path, kind, tuple(sorted(cfg.items())))
    if key not in _CACHE:
        m = build(kind, **cfg)
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and not any(
                k.startswith(("stem", "body", "foot", "head")) for k in sd):
            sd = sd["state_dict"]
        m.load_state_dict(sd); m.eval()
        _CACHE[key] = m
    return _CACHE[key]


def _logits(m, obs, mask):
    with torch.no_grad():
        lg = m({"is_training": False, "obs": {
            "observation": torch.from_numpy(np.ascontiguousarray(obs)),
            "action_mask": torch.from_numpy(np.ascontiguousarray(mask))}})
    return lg.numpy().flatten()


def _greedy_lg(m):
    def fn(obs, mask, return_logits=False):
        lg = _logits(m, obs, mask)
        return lg if return_logits else [int(lg.argmax())]
    return fn


class LogSim(Sim):
    """Logs (seat, win_prob) at each discard turn into self.records."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.records = []  # (seat, p)

    def _ask(self, seat):
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        lg = self.policies[seat](obs[None, :], mask[None, :], return_logits=True)
        a = int(lg.argmax())
        if not mask[a]:
            a = int(np.flatnonzero(mask)[0])
        # log win-prob ONLY on genuine discard turns (a Play action), matching where the controller fires
        if PLAY <= a < PLAY + 34 and self.cai is not None:
            try:
                p = WP.win_prob(self.cai[seat])
                self.records.append((seat, p))
            except Exception:
                pass
        if seat in self.learner_seats:
            self.traj[seat].append((obs, mask, a))
        return a


def _work(arg):
    seed, cand, ck, ccfg = arg
    m = _load(cand, ck, ccfg)
    f = _greedy_lg(m)
    sim = LogSim([f, f, f, f], seed=seed, quan=0, learner_seats=[], cnn=True)
    sim.play()
    won = [1 if sim.scores[s] > 0 else 0 for s in range(4)]
    return [(p, won[seat]) for (seat, p) in sim.records]


def _auc(ps, ys):
    # rank-based AUC
    ps = np.asarray(ps); ys = np.asarray(ys)
    npos = ys.sum(); nneg = len(ys) - npos
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(ps, kind="mergesort")
    ranks = np.empty(len(ps)); ranks[order] = np.arange(1, len(ps) + 1)
    # average ranks for ties
    sp = ps[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            r = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = r
        i = j + 1
    sum_pos = ranks[ys == 1].sum()
    return float((sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True); ap.add_argument("--cand-kind", default="resbn")
    ap.add_argument("--cand-cfg", default="channels=128,blocks=40")
    ap.add_argument("--games", type=int, default=800)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=200000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = {}
    for kv in a.cand_cfg.split(","):
        k, v = kv.split("="); cfg[k] = int(v)
    args = [(a.seed0 + i, a.cand, a.cand_kind, cfg) for i in range(a.games)]
    t0 = time.time()
    with mp.Pool(a.workers) as p:
        res = p.map(_work, args, chunksize=2)
    recs = [r for sub in res for r in sub]
    if not recs:
        json.dump({"n": 0, "FAIL": "n=0 records"}, open(a.out, "w"))
        print("FAIL: n=0 records"); sys.exit(2)
    ps = np.array([r[0] for r in recs]); ys = np.array([r[1] for r in recs])
    brier = float(np.mean((ps - ys) ** 2))
    base_rate = float(ys.mean())
    brier_base = float(np.mean((base_rate - ys) ** 2))  # predicting the constant base rate
    auc = _auc(ps, ys)
    # reliability table (10 bins)
    bins = np.linspace(0, 1, 11)
    rel = []
    for b in range(10):
        m = (ps >= bins[b]) & (ps < bins[b + 1] if b < 9 else ps <= bins[b + 1])
        if m.sum() > 0:
            rel.append({"bin": [round(bins[b], 1), round(bins[b + 1], 1)],
                        "n": int(m.sum()), "pred_mean": round(float(ps[m].mean()), 4),
                        "emp_winrate": round(float(ys[m].mean()), 4)})
    out = dict(cand=os.path.basename(a.cand), games=a.games, n_decisions=len(recs),
               base_winrate=round(base_rate, 4), auc=round(auc, 4) if auc is not None else None,
               brier=round(brier, 4), brier_baserate=round(brier_base, 4),
               brier_skill=round(1 - brier / brier_base, 4) if brier_base > 0 else None,
               pred_mean=round(float(ps.mean()), 4), pred_std=round(float(ps.std()), 4),
               reliability=rel, seconds=round(time.time() - t0, 1), seed0=a.seed0)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
