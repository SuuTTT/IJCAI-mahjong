"""e6_switcher_eval.py — E6-P2(b) evaluate the opponent-conditional switcher.

Policy: kdens3 for discard-turns < T; at turn T one estimator call; if it says
WEAK, aug_s0's softmax drives all remaining decisions, else stay kdens3.
Evaluated at seat 0 on ALL 4 Phase-1 fields, n games/field, SAME seed blocks as
the Phase-1 kdens3 cells (7000000 + field_idx*100000) for pairing.

VERDICT (vs FIELD_RESPONSE.json cells): does the switcher sit on the UPPER
ENVELOPE — match max(kdens3, aug_s0) score on every field within 95% CI?
(Unpaired CI on the diff: Phase-1 stored only aggregates. Switcher seeds are
paired with the kdens3 cells, so vs-kdens3 diffs are conservative.)

  python3 e6_switcher_eval.py --ngames 2000 --workers 110 \
      --est ckpt/e6/e6_est_t8.pt --T 8 --out results/E6_SWITCHER.json
T/est default from results/E6_ESTIMATOR.json chosen_T.
"""
import os, sys, json, argparse, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
from e6_switch_common import (FIELDS, FIELD_ORDER, KD_PATHS, AUG_PATH, CLASSES,
                              SwitchSim, game_stats, load_net, load_estimator,
                              policy_fn, preload_all)

EST = {"path": None, "T": None}


def _work(arg):
    seed, field = arg
    est = load_estimator(EST["path"])
    kd = [load_net(p) for p in KD_PATHS]
    aug = load_net(AUG_PATH)
    pols = [None] + [policy_fn([p]) for p in FIELDS[field]]
    sim = SwitchSim(pols, target=0, kd_models=kd, aug_model=aug, est=est,
                    T=EST["T"], seed=seed, quan=0, learner_seats=[], cnn=True)
    sim.play()
    rank, score, dealin, win, draw = game_stats(sim, 0)
    return rank, score, dealin, win, draw, (-1 if sim.pred is None else sim.pred), sim.turn


def _stats(vals):
    a = np.asarray(vals, dtype=np.float64)
    n = len(a); m = float(a.mean()); sd = float(a.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    return dict(mean=round(m, 4), se=round(se, 4),
                ci95=[round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)], n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngames", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=110)
    ap.add_argument("--seed0", type=int, default=7000000)   # Phase-1 kdens3 cells
    ap.add_argument("--est", default=None)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--estimator_json", default="results/E6_ESTIMATOR.json")
    ap.add_argument("--phase1", default="results/FIELD_RESPONSE.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.T is None or a.est is None:
        ej = json.load(open(a.estimator_json))
        a.T = a.T or int(ej["chosen_T"])
        a.est = a.est or ej["turns"][str(a.T)]["ckpt"]
    EST["path"], EST["T"] = a.est, a.T
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    preload_all(); load_estimator(a.est)
    p1 = json.load(open(a.phase1))["cells"]
    cells = {}
    meta = dict(design="switcher seat0 (kdens3 -> estimator at discard-turn T -> "
                       "aug_s0 iff pred=weak) vs Phase-1 fields; same seeds as "
                       "Phase-1 kdens3 cells", T=a.T, estimator=os.path.abspath(a.est),
                ngames_per_field=a.ngames, seed0=a.seed0, started=time.strftime("%F %T"))
    for fi, field in enumerate(FIELD_ORDER):
        t0 = time.time()
        s0 = a.seed0 + fi * 100000
        args = [(s0 + i, field) for i in range(a.ngames)]
        with mp.Pool(a.workers) as p:
            res = p.map(_work, args, chunksize=4)
        preds = [r[5] for r in res]
        n = len(res)
        cells[field] = dict(
            field=field, n=n, seed0=s0,
            rank=_stats([r[0] for r in res]), score=_stats([r[1] for r in res]),
            dealin_rate=round(sum(r[2] for r in res) / n, 4),
            win_rate=round(sum(r[3] for r in res) / n, 4),
            draw_rate=round(sum(r[4] for r in res) / n, 4),
            switch_rate=round(sum(1 for x in preds if x == 0) / n, 4),
            pred_dist={CLASSES[c]: round(sum(1 for x in preds if x == c) / n, 4)
                       for c in range(3)},
            no_pred_rate=round(sum(1 for x in preds if x < 0) / n, 4),  # game ended before T
            seconds=round(time.time() - t0, 1))
        print(f"FIELD {field}: score={cells[field]['score']['mean']} "
              f"rank={cells[field]['rank']['mean']} switch={cells[field]['switch_rate']} "
              f"({cells[field]['seconds']}s)", flush=True)
        with open(a.out, "w") as f:
            json.dump(dict(meta=meta, cells=cells, done=fi + 1, total=4), f, indent=2)

    # ---- envelope verdict vs Phase-1 kdens3 / aug_s0 ----
    env = {"per_field": {}, "on_upper_envelope": True}
    for field in FIELD_ORDER:
        sw = cells[field]["score"]
        comp = {}
        for pol in ("kdens3", "aug_s0"):
            c = p1[f"{pol}|{field}"]["score"]
            d = sw["mean"] - c["mean"]
            se = math.sqrt(sw["se"] ** 2 + c["se"] ** 2)
            comp[pol] = dict(base_mean=c["mean"], diff=round(d, 4),
                             ci95=[round(d - 1.96 * se, 4), round(d + 1.96 * se, 4)],
                             significant=bool(abs(d) > 1.96 * se))
        best = max(("kdens3", "aug_s0"), key=lambda p: p1[f"{p}|{field}"]["score"]["mean"])
        on_env = comp[best]["ci95"][1] >= 0 or comp[best]["diff"] >= 0
        env["per_field"][field] = dict(best_base=best, vs=comp,
                                       on_envelope=bool(on_env))
        if not on_env:
            env["on_upper_envelope"] = False
    # the two decisive checks
    env["beats_kdens3_on_weak"] = env["per_field"]["weak"]["vs"]["kdens3"]
    env["matches_kdens3_on_champion"] = env["per_field"]["champion"]["vs"]["kdens3"]
    out = dict(meta=meta, cells=cells, envelope=env, done=4, total=4,
               finished=time.strftime("%F %T"))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("VERDICT on_upper_envelope =", env["on_upper_envelope"], flush=True)
    print(json.dumps(env["per_field"], indent=1), flush=True)


if __name__ == "__main__":
    main()
