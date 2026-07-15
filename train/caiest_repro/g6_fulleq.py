"""
g6_fulleq.py -- END-TO-END bit-for-bit equivalence: full-game per-(seed,seat) placement
of s6's ORIGINAL PIMCVSim vs g6's EXACT-mode G6Sim. Same seeds, same determinization RNG.
Parallelized: one task = (engine, seed, seat) -> one game -> placement.
"""
import os, sys, json, argparse, time
sys.path.insert(0, "/root/caiest_repro")
import numpy as np, multiprocessing as mp

_CFG = {}


def _init(cfg):
    import torch; torch.set_num_threads(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU everywhere (exact)
    import s6_pimc_vcut as s6, g6_engine
    g6_engine.init_infer(mode="exact", null=cfg["null"], device="cpu")  # sets s6._G kd/vh/src
    s6._G.update(dict(null=cfg["null"], true_state=False,
                      n_worlds=cfg["n_worlds"], k_cutoff=cfg["k_cutoff"]))
    _CFG.update(cfg)


def _game(arg):
    engine, seed, cs = arg
    import s6_pimc_vcut as s6, g6_engine
    from s6_pimc_vcut import _placement
    if engine == "s6":
        sim = s6.PIMCVSim([None] * 4, seed=seed, quan=0, learner_seats=[], cnn=True)
    else:
        sim = g6_engine.G6Sim([None] * 4, seed=seed, quan=0, learner_seats=[], cnn=True)
    sim.search_seat = cs
    sim.null = _CFG["null"]; sim.true_state = False
    sim.n_worlds = _CFG["n_worlds"]; sim.k_cutoff = _CFG["k_cutoff"]
    sim._rng = np.random.RandomState((seed * 4 + cs) % (2**31 - 1))
    sim.play()
    return engine, seed, cs, _placement(sim.scores, cs), list(sim.scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0", type=int, default=9800000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--n_worlds", type=int, default=8)
    ap.add_argument("--k_cutoff", type=int, default=6)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--out", default="results/G6_FULLEQ.json")
    a = ap.parse_args()
    cfg = dict(null=False, n_worlds=a.n_worlds, k_cutoff=a.k_cutoff)
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    tasks = [(e, s, cs) for e in ("s6", "g6") for s in seeds for cs in range(4)]
    t0 = time.time()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(a.procs, initializer=_init, initargs=(cfg,)) as p:
        res = p.map(_game, tasks, chunksize=1)
    D = {}
    for engine, seed, cs, pl, sc in res:
        D[(engine, seed, cs)] = (pl, tuple(sc))
    rows = []; mism = 0
    for seed in seeds:
        for cs in range(4):
            s6v = D[("s6", seed, cs)]; g6v = D[("g6", seed, cs)]
            ok = (s6v == g6v)
            if not ok: mism += 1
            rows.append(dict(seed=seed, seat=cs, s6_pl=s6v[0], g6_pl=g6v[0],
                             s6_scores=list(s6v[1]), g6_scores=list(g6v[1]), match=ok))
    out = dict(test="full-game bit-for-bit s6 vs g6-exact", n_worlds=a.n_worlds,
               k_cutoff=a.k_cutoff, seeds=seeds, n_games=len(seeds) * 4,
               mismatches=mism, all_match=(mism == 0),
               s6_mean_placement=round(float(np.mean([r["s6_pl"] for r in rows])), 4),
               g6_mean_placement=round(float(np.mean([r["g6_pl"] for r in rows])), 4),
               rows=rows, seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("PER-SEED/SEAT:")
    for r in rows:
        print(f"  seed {r['seed']} seat {r['seat']}: s6={r['s6_pl']:.3f} g6={r['g6_pl']:.3f} "
              f"match={r['match']}")
    print("FULLEQ", "PASS" if mism == 0 else f"FAIL ({mism} mismatches)")


if __name__ == "__main__":
    main()
