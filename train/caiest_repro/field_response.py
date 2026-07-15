"""
field_response.py — E6 Phase 1: FIELD-RESPONSE CURVES.

Question: do policy payoffs depend on opponent-field strength (do curves CROSS
between styles)? Motivated by the IJCAI final inversion (2nd-in-16-field bot
collapsed to -9129 vs the top-3 field).

Design: policy at seat 0, a chosen 3-opponent FIELD at seats 1-3.
  Policies: kdens3 (kd_s0/1/2 mean-softmax), aug_s0 (single),
            f2mix  (armb_s0/1/2 = mix_beta0.3_botsall corpus-mix, mean-softmax)
  Fields:   WEAK  = kdf_s0, kdf_s1, kdf_s2      (field-adapted weak clones)
            MIXED = aug_s0, kdf_s0, kdf_s1
            STRONG= aug_s0 x3
            CHAMP = kd_s0, kd_s1, kd_s2         (kdens3 students as 3 singles)
Per game: rank among 4 by raw score (ties averaged), raw seat-0 score,
deal-in (rong charged to seat 0), win (seat 0 rong/zimo), draw.
Seeds disjoint per cell. Ensemble rule = EXACT deploy rule from
e12_score_gate.py (mean softmax over legal, log).

  python3 field_response.py --ngames 2000 --workers 80 \
      --out results/FIELD_RESPONSE.json
"""
import os, sys, json, argparse, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, multiprocessing as mp
from sim_cnn import Sim
from models_explore import build

torch.set_num_threads(1)
_CACHE = {}
CKPT = "/root/caiest_repro/ckpt"

POLICIES = {  # seat-0 candidates; list>1 -> deploy mean-softmax ensemble
    "kdens3": [f"{CKPT}/kd/kd_128x40_s{i}.pkl" for i in range(3)],
    "aug_s0": [f"{CKPT}/aug/aug_128x40_s0.pkl"],
    "f2mix":  [f"{CKPT}/f2/armb_s{i}.pkl" for i in range(3)],
}
FIELDS = {  # 3 opponents, each a SINGLE net at one seat (1,2,3 in order)
    "weak":  [f"{CKPT}/kdfield/kdf_s0.pkl", f"{CKPT}/kdfield/kdf_s1.pkl", f"{CKPT}/kdfield/kdf_s2.pkl"],
    "mixed": [f"{CKPT}/aug/aug_128x40_s0.pkl", f"{CKPT}/kdfield/kdf_s0.pkl", f"{CKPT}/kdfield/kdf_s1.pkl"],
    "strong": [f"{CKPT}/aug/aug_128x40_s0.pkl"] * 3,
    "champion": [f"{CKPT}/kd/kd_128x40_s{i}.pkl" for i in range(3)],
}
FIELD_ORDER = ["weak", "mixed", "strong", "champion"]
POL_ORDER = ["kdens3", "aug_s0", "f2mix"]


def _load(path):
    if path not in _CACHE:
        m = build("resbn_fused", channels=128, blocks=40)
        m.load_state_dict(torch.load(path, map_location="cpu")); m.eval()
        _CACHE[path] = m
    return _CACHE[path]


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


def _policy(paths):
    models = [_load(p) for p in paths]
    return _ens_fn(models) if len(models) > 1 else _single_fn(models[0])


class FRSim(Sim):
    """PSim._ask from e12_score_gate + DISim deal-in/win counting from lever1_eval."""
    def __init__(self, *a, target=0, **k):
        super().__init__(*a, **k)
        self.target = target; self.dealins = 0; self.wins = 0

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

    def _score_rong(self, w, src, f):
        super()._score_rong(w, src, f)
        if src == self.target: self.dealins += 1
        if w == self.target: self.wins += 1

    def _score_selfdraw(self, w, f):
        super()._score_selfdraw(w, f)
        if w == self.target: self.wins += 1


def _work(arg):
    seed, pol_paths, field_paths = arg
    pols = [_policy(pol_paths)] + [_policy([p]) for p in field_paths]
    sim = FRSim(pols, seed=seed, quan=0, learner_seats=[], cnn=True, target=0)
    sim.play()
    sc = sim.scores; c = sc[0]
    greater = sum(1 for j in range(4) if sc[j] > c)
    equal = sum(1 for j in range(4) if sc[j] == c)  # includes self
    rank = greater + (equal + 1) / 2.0              # 1=best, 4=worst, ties averaged
    win = int(sim.wins > 0)
    dealin = int(sim.dealins > 0)
    draw = int(all(s == 0 for s in sc))             # no one scored
    return rank, float(c), dealin, win, draw


def _stats(vals):
    a = np.asarray(vals, dtype=np.float64)
    n = len(a); m = float(a.mean()); sd = float(a.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    return dict(mean=round(m, 4), se=round(se, 4),
                ci95=[round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)], n=n)


def crossing_verdict(cells):
    """For each policy pair + metric, does the ordering FLIP between fields
    (weak->strong and weak->champion), with both diffs' 95% CIs excluding 0?"""
    pairs = [("kdens3", "aug_s0"), ("kdens3", "f2mix"), ("aug_s0", "f2mix")]
    out = {"pairs": {}, "any_significant_crossing": False, "crossings": []}
    for A, B in pairs:
        pr = {}
        for metric in ("score", "rank"):
            per_field = {}
            for f in FIELD_ORDER:
                ca, cb = cells.get(f"{A}|{f}"), cells.get(f"{B}|{f}")
                if not ca or not cb:
                    continue
                d = ca[metric]["mean"] - cb[metric]["mean"]
                se = math.sqrt(ca[metric]["se"] ** 2 + cb[metric]["se"] ** 2)
                per_field[f] = dict(diff=round(d, 4),
                                    ci95=[round(d - 1.96 * se, 4), round(d + 1.96 * se, 4)],
                                    significant=bool(abs(d) > 1.96 * se))
            flips = {}
            for f2 in ("mixed", "strong", "champion"):
                if "weak" not in per_field or f2 not in per_field:
                    continue
                a, b = per_field["weak"], per_field[f2]
                sign_flip = (a["diff"] * b["diff"] < 0)
                sig = sign_flip and a["significant"] and b["significant"]
                flips[f"weak_vs_{f2}"] = dict(
                    sign_flip=bool(sign_flip), significant_crossing=bool(sig),
                    magnitude=round(abs(a["diff"]) + abs(b["diff"]), 4) if sign_flip else 0.0)
                if sig:
                    out["any_significant_crossing"] = True
                    out["crossings"].append(dict(pair=f"{A} vs {B}", metric=metric,
                                                 fields=f"weak->{f2}",
                                                 diff_weak=a["diff"], diff_other=b["diff"]))
            pr[metric] = dict(per_field=per_field, flips=flips)
        out["pairs"][f"{A} vs {B}"] = pr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngames", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=80)
    ap.add_argument("--seed0", type=int, default=7000000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cells = {}
    meta = dict(design="policy seat0 vs 3-opponent field seats1-3; single round quan=0; "
                       "ensemble=deploy mean-softmax-over-legal; ranks tie-averaged",
                policies={k: [os.path.basename(p) for p in v] for k, v in POLICIES.items()},
                fields={k: [os.path.basename(p) for p in v] for k, v in FIELDS.items()},
                ngames_per_cell=a.ngames, seed0=a.seed0, started=time.strftime("%F %T"))
    cell_idx = 0
    for pol in POL_ORDER:
        for field in FIELD_ORDER:
            t0 = time.time()
            s0 = a.seed0 + cell_idx * 100000  # disjoint seed block per cell
            args = [(s0 + i, POLICIES[pol], FIELDS[field]) for i in range(a.ngames)]
            with mp.Pool(a.workers) as p:
                res = p.map(_work, args, chunksize=4)
            ranks = [r[0] for r in res]; scores = [r[1] for r in res]
            cells[f"{pol}|{field}"] = dict(
                policy=pol, field=field, n=len(res), seed0=s0,
                rank=_stats(ranks), score=_stats(scores),
                dealin_rate=round(sum(r[2] for r in res) / len(res), 4),
                win_rate=round(sum(r[3] for r in res) / len(res), 4),
                draw_rate=round(sum(r[4] for r in res) / len(res), 4),
                seconds=round(time.time() - t0, 1))
            cell_idx += 1
            out = dict(meta=meta, cells=cells, done=cell_idx, total=12)
            with open(a.out, "w") as f:
                json.dump(out, f, indent=2)
            print(f"CELL {cell_idx}/12 {pol}|{field}: rank={cells[f'{pol}|{field}']['rank']['mean']} "
                  f"score={cells[f'{pol}|{field}']['score']['mean']} "
                  f"dealin={cells[f'{pol}|{field}']['dealin_rate']} "
                  f"win={cells[f'{pol}|{field}']['win_rate']} "
                  f"({cells[f'{pol}|{field}']['seconds']}s)", flush=True)
    verdict = crossing_verdict(cells)
    out = dict(meta=meta, cells=cells, crossing=verdict, done=12, total=12,
               finished=time.strftime("%F %T"))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("VERDICT any_significant_crossing =", verdict["any_significant_crossing"], flush=True)
    print(json.dumps(verdict["crossings"], indent=1), flush=True)


if __name__ == "__main__":
    main()
