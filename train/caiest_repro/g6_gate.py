"""
g6_gate.py -- GPU-batched PIMC value-cutoff confirming gate (drop-in for s6, ~100x faster).

Uses g6_engine.G6Sim: the game logic is a line-for-line port of s6, but the N-worlds x
candidates rollouts at each search decision run in LOCKSTEP with ONE batched GPU forward
per step (kd ensemble policy + value-head leaves).

Parallelism: many worker processes spread across GPUs (default 0,1,2). Each process runs
whole games sequentially; its own rollout forwards batch on its assigned GPU.

Identical experiment protocol to s6: seeds 9_800_000 + b*3000 + [0..seeds), 4 seats/seed,
block-mean placement + t-CI. Determinization RNG RandomState((seed*4+cs)%(2**31-1)).

  python3 g6_gate.py --mode gpu --blocks 0 --seeds 8 --n_worlds 20 --k_cutoff 6 \
      --gpus 0,1,2 --procs 24 --out results/PIMC_GATE_BATCHED.json
  python3 g6_gate.py --mode gpu --null --blocks 0 --seeds 40 --gpus 0 --procs 8 --out /tmp/null.json
"""
import os, sys, json, argparse, time, math
sys.path.insert(0, "/root/caiest_repro")
import numpy as np, multiprocessing as mp

_CFG = {}


def _init_worker(cfg, gpus):
    import torch
    torch.set_num_threads(1)
    wid = (mp.current_process()._identity[0] - 1) if mp.current_process()._identity else 0
    gpu = gpus[wid % len(gpus)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import g6_engine
    dev = "cpu" if cfg["mode"] == "exact" else "cuda:0"
    g6_engine.init_infer(mode=cfg["mode"], null=cfg["null"], device=dev,
                         want_belief=cfg.get("belief", False),
                         want_place=(cfg.get("leaf", "score") == "placement"))
    _CFG.update(cfg)
    _CFG["_engine"] = g6_engine


def _work(arg):
    block, seed = arg
    import g6_engine
    from g6_engine import G6Sim
    from s6_pimc_vcut import _placement
    psum = 0.0; ov = dec = skip = good = bad = 0
    for cs in range(4):
        sim = G6Sim([None] * 4, seed=seed, quan=0, learner_seats=[], cnn=True)
        sim.search_seat = cs
        sim.null = _CFG.get("null", False)
        sim.true_state = _CFG.get("true_state", False)
        sim.n_worlds = _CFG.get("n_worlds", 1)
        sim.k_cutoff = _CFG.get("k_cutoff", 6)
        sim.belief = _CFG.get("belief", False)
        sim.leaf = _CFG.get("leaf", "score")
        sim._rng = np.random.RandomState((seed * 4 + cs) % (2**31 - 1))
        sim.play()
        psum += _placement(sim.scores, cs)
        ov += getattr(sim, "_override", 0); dec += getattr(sim, "_decisions", 0)
        skip += getattr(sim, "_skipped", 0)
        good += getattr(sim, "_good_world", 0); bad += getattr(sim, "_bad_world", 0)
    return block, seed, psum, ov, dec, skip, good, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="gpu", choices=["gpu", "exact"])
    ap.add_argument("--blocks", default="0")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--n_worlds", type=int, default=20)
    ap.add_argument("--k_cutoff", type=int, default=6)
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--true_state", action="store_true")
    ap.add_argument("--belief", action="store_true")
    ap.add_argument("--leaf", default="score", choices=["score", "placement"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    blocks = [int(x) for x in a.blocks.split(",")]
    gpus = [int(x) for x in a.gpus.split(",")]
    tasks = []; seedmap = {}
    for b in blocks:
        s0 = 9_800_000 + b * 3000
        seedmap[str(b)] = [s0, s0 + a.seeds]
        tasks += [(b, s) for s in range(s0, s0 + a.seeds)]
    cfg = dict(mode=a.mode, null=a.null, true_state=a.true_state,
               n_worlds=a.n_worlds, k_cutoff=a.k_cutoff, belief=a.belief, leaf=a.leaf)
    t0 = time.time()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(a.procs, initializer=_init_worker, initargs=(cfg, gpus)) as p:
        res = p.map(_work, tasks, chunksize=1)
    per_block = {b: [] for b in blocks}
    per_seed = {}
    tot_ov = tot_dec = tot_skip = tot_good = tot_bad = 0
    for b, s, psum, ov, dec, skip, good, bad in res:
        per_block[b].append(psum / 4.0); per_seed[f"{b}:{s}"] = round(psum / 4.0, 4)
        tot_ov += ov; tot_dec += dec; tot_skip += skip; tot_good += good; tot_bad += bad
    block_means = {b: float(np.mean(per_block[b])) for b in blocks}
    bm = np.array([block_means[b] for b in blocks], dtype=np.float64)
    n = len(bm); block_mean = float(bm.mean())
    block_sd = float(bm.std(ddof=1)) if n > 1 else 0.0
    se = block_sd / math.sqrt(n) if n > 1 else 0.0
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
             9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160}.get(n, 1.96)
    ci = tcrit * se; lo, hi = block_mean - ci, block_mean + ci
    if a.null:
        verdict = ("NULL-CAL OK (2.5000)" if abs(block_mean - 2.5) < 1e-6
                   else f"NULL-CAL BROKEN (dev {block_mean-2.5:+.6f})")
    elif a.true_state:
        verdict = f"TRUE-STATE ceiling = {block_mean:.4f}"
    else:
        tag = f"N={a.n_worlds} K={a.k_cutoff} belief={a.belief} leaf={a.leaf}"
        if lo > 2.5:
            verdict = f"PIMC-vcut {tag} CONFIRMS >2.5 (CI lower {lo:.4f}) -- deployable champion-beater"
        elif hi < 2.5:
            verdict = f"PIMC-vcut {tag} LOSES (CI<2.5)"
        else:
            verdict = f"PIMC-vcut {tag} TIES (CI spans 2.5)"
    out = dict(experiment="PIMC value-cutoff discard search vs kdens3 (GPU-batched g6)",
               engine="g6_engine lockstep batched", infer_mode=a.mode,
               belief=bool(a.belief), leaf=a.leaf,
               null_run=bool(a.null), true_state=bool(a.true_state),
               n_worlds=a.n_worlds, k_cutoff=a.k_cutoff, blocks=blocks,
               seeds_per_block=a.seeds, seed_ranges=seedmap, n_blocks=n,
               n_seeds=len(tasks), n_games=len(tasks) * 4, gpus=gpus, procs=a.procs,
               block_means={str(b): round(block_means[b], 4) for b in blocks},
               block_mean_placement=round(block_mean, 4), block_sd=round(block_sd, 4),
               ci95=[round(lo, 4), round(hi, 4)],
               override_fraction=round(tot_ov / max(1, tot_dec), 4),
               n_search_decisions=tot_dec, n_overrides=tot_ov,
               n_worlds_ok=tot_good, n_worlds_rejected=tot_bad,
               reject_fraction=round(tot_bad / max(1, tot_good + tot_bad), 6),
               verdict=verdict, seconds=round(time.time() - t0, 1),
               games_per_hour=round(len(tasks) * 4 / ((time.time() - t0) / 3600.0), 1),
               per_seed=per_seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_seed"}, indent=1))


if __name__ == "__main__":
    main()
