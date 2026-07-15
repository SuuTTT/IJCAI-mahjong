"""e6_switcher_gate.py — E6-P2(c) standard e12 duplicate gate: switcher vs kdens3.

Same instrument as e12_score_gate.py (placement points + raw duplicate score,
4 rotations per seed), cand = the switcher (kdens3 + estimator-at-turn-T +
aug_s0-if-weak), ref = plain kdens3 ensemble at the other 3 seats. The ref
field IS the Phase-1 champion field, so a correct estimator should rarely
switch; the gate confirms no regression on the standard instrument.

Block mode (12 blocks x 500 seeds x 4 rotations = 2000 games/block, matching
the historical gate protocol):
  python3 e6_switcher_gate.py --seeds 500 --seed0 8000000 --out results/e6_gate/e6sw_b0.json
Aggregate mode:
  python3 e6_switcher_gate.py --agg 'results/e6_gate/e6sw_b*.json' --aggout results/E6_SWITCHER_GATE.json
"""
import os, sys, json, argparse, time, glob, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
from e6_switch_common import (KD_PATHS, AUG_PATH, SwitchSim, load_net,
                              load_estimator, policy_fn, preload_all)

EST = {"path": None, "T": None}


def _work(arg):
    seed = arg
    est = load_estimator(EST["path"])
    kd = [load_net(p) for p in KD_PATHS]
    aug = load_net(AUG_PATH)
    ref = policy_fn(KD_PATHS)
    placement_sum = 0.0; cand_sc = 0.0; ref_sc = 0.0; nswitch = 0
    for cs in range(4):
        pols = [ref] * 4
        sim = SwitchSim(pols, target=cs, kd_models=kd, aug_model=aug, est=est,
                        T=EST["T"], seed=seed, quan=0, learner_seats=[], cnn=True)
        sim.play()
        sc = sim.scores; c = sc[cs]
        greater = sum(1 for j in range(4) if sc[j] > c)
        equal = sum(1 for j in range(4) if sc[j] == c)
        placement_sum += 5.0 - (greater + (equal + 1) / 2.0)
        cand_sc += c
        ref_sc += sum(sc[j] for j in range(4) if j != cs)
        if sim.pred == 0: nswitch += 1
    return placement_sum, cand_sc, ref_sc, nswitch


def aggregate(pattern, aggout):
    blocks = [json.load(open(p)) for p in sorted(glob.glob(pattern))]
    pl = [b["placement_pts"] for b in blocks]
    sd = [b["score_diff"] for b in blocks]
    sw = [b["switch_rate"] for b in blocks]
    n = len(blocks)
    pm = float(np.mean(pl)); pse = float(np.std(pl, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    sm = float(np.mean(sd)); sse = float(np.std(sd, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    out = dict(design="e12 duplicate gate, cand=switcher(T from estimator) ref=kdens3; "
                      "12 blocks x 2000 games, fresh seeds 8000000+",
               blocks=n, games=sum(b["games"] for b in blocks),
               placement_blocks=[round(x, 4) for x in pl],
               placement_mean=round(pm, 4), placement_se=round(pse, 4),
               placement_ci95=[round(pm - 1.96 * pse, 4), round(pm + 1.96 * pse, 4)],
               score_diff_blocks=[round(x, 4) for x in sd],
               score_diff_mean=round(sm, 4), score_diff_se=round(sse, 4),
               score_diff_ci95=[round(sm - 1.96 * sse, 4), round(sm + 1.96 * sse, 4)],
               switch_rate_mean=round(float(np.mean(sw)), 4),
               no_regression=bool(pm + 1.96 * pse >= 2.5 and sm + 1.96 * sse >= 0),
               T=blocks[0].get("T"), finished=time.strftime("%F %T"))
    with open(aggout, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--workers", type=int, default=110)
    ap.add_argument("--seed0", type=int, default=8000000)
    ap.add_argument("--est", default=None)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--estimator_json", default="results/E6_ESTIMATOR.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--agg", default=None)
    ap.add_argument("--aggout", default="results/E6_SWITCHER_GATE.json")
    a = ap.parse_args()
    if a.agg:
        return aggregate(a.agg, a.aggout)
    if a.T is None or a.est is None:
        ej = json.load(open(a.estimator_json))
        a.T = a.T or int(ej["chosen_T"])
        a.est = a.est or ej["turns"][str(a.T)]["ckpt"]
    EST["path"], EST["T"] = a.est, a.T
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    preload_all(); load_estimator(a.est)
    args = [a.seed0 + i for i in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers) as p:
        res = p.map(_work, args, chunksize=1)
    ngames = len(res) * 4
    pts = sum(r[0] for r in res) / ngames
    csc = sum(r[1] for r in res) / ngames
    rsc = sum(r[2] for r in res) / (3 * ngames)
    swr = sum(r[3] for r in res) / ngames
    out = dict(cand=f"switcher(T={a.T}, est={os.path.basename(a.est)})",
               ref=[os.path.basename(p) for p in KD_PATHS],
               rule="mean-softmax-over-legal (deploy ensemble_infer); switcher cand seat",
               games=ngames, placement_pts=round(pts, 4),
               cand_score_mean=round(csc, 4), ref_score_mean=round(rsc, 4),
               score_diff=round(csc - rsc, 4), switch_rate=round(swr, 4),
               T=a.T, seconds=round(time.time() - t0, 1), seed0=a.seed0)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
