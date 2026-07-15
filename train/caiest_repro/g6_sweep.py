"""
g6_sweep.py -- PIMC config sweep over {belief} x {leaf} x {K} x {N}, using the GPU-batched
g6 engine. Runs cells cheap->expensive, writes results/PIMC_SWEEP.json INCREMENTALLY after
each cell (so promising cells are visible early). Each cell = `blocks` x `seeds` seeds
(x4 seats), block-mean placement + t-CI, override & reject fractions.

  python3 g6_sweep.py --beliefs 0,1 --leaves score,placement --Ks 6,12 --Ns 20 \
     --blocks 6 --seeds 15 --gpus 0,1,2 --procs 48 --out results/PIMC_SWEEP.json
"""
import os, sys, json, argparse, time, math, itertools
sys.path.insert(0, "/root/caiest_repro")
import numpy as np, multiprocessing as mp
import g6_gate

TCRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
         9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160}


def run_cell(belief, leaf, K, N, blocks, seeds, gpus, procs):
    tasks = []
    for b in blocks:
        s0 = 9_800_000 + b * 3000
        tasks += [(b, s) for s in range(s0, s0 + seeds)]
    cfg = dict(mode="gpu", null=False, true_state=False, n_worlds=N, k_cutoff=K,
               belief=bool(belief), leaf=leaf)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs, initializer=g6_gate._init_worker, initargs=(cfg, gpus)) as p:
        res = p.map(g6_gate._work, tasks, chunksize=1)
    per_block = {b: [] for b in blocks}
    tot_ov = tot_dec = tot_good = tot_bad = 0
    for b, s, psum, ov, dec, skip, good, bad in res:
        per_block[b].append(psum / 4.0)
        tot_ov += ov; tot_dec += dec; tot_good += good; tot_bad += bad
    bm = np.array([np.mean(per_block[b]) for b in blocks], dtype=np.float64)
    n = len(bm); mean = float(bm.mean())
    sd = float(bm.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    ci = TCRIT.get(n, 1.96) * se
    return dict(belief=bool(belief), leaf=leaf, K=K, N=N,
                blocks=len(blocks), seeds_per_block=seeds, n_games=len(tasks) * 4,
                block_mean_placement=round(mean, 4), block_sd=round(sd, 4),
                ci95=[round(mean - ci, 4), round(mean + ci, 4)],
                ci_lower=round(mean - ci, 4),
                clears_2p5=bool(mean - ci > 2.5),
                override_fraction=round(tot_ov / max(1, tot_dec), 4),
                reject_fraction=round(tot_bad / max(1, tot_good + tot_bad), 6),
                n_decisions=tot_dec, seconds=round(time.time() - t0, 1),
                games_per_hour=round(len(tasks) * 4 / ((time.time() - t0) / 3600.0), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beliefs", default="0,1")
    ap.add_argument("--leaves", default="score,placement")
    ap.add_argument("--Ks", default="6,12")
    ap.add_argument("--Ns", default="20")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--out", default="results/PIMC_SWEEP.json")
    a = ap.parse_args()
    beliefs = [int(x) for x in a.beliefs.split(",")]
    leaves = a.leaves.split(",")
    Ks = [int(x) for x in a.Ks.split(",")]
    Ns = [int(x) for x in a.Ns.split(",")]
    blocks = list(range(a.blocks))
    gpus = [int(x) for x in a.gpus.split(",")]
    # cells cheap -> expensive (by N*K)
    cells = list(itertools.product(beliefs, leaves, Ks, Ns))
    cells.sort(key=lambda c: c[2] * c[3])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    results = []
    meta = dict(sweep="belief x leaf x K x N", blocks=a.blocks, seeds_per_block=a.seeds,
                gpus=gpus, procs=a.procs, kdens3_baseline=2.5, started=time.strftime("%H:%M:%S"))
    for (belief, leaf, K, N) in cells:
        print(f"\n=== CELL belief={belief} leaf={leaf} K={K} N={N} "
              f"({a.blocks}x{a.seeds}) ===", flush=True)
        r = run_cell(belief, leaf, K, N, blocks, a.seeds, gpus, a.procs)
        results.append(r)
        print(f"  -> placement={r['block_mean_placement']} CI={r['ci95']} "
              f"clears2.5={r['clears_2p5']} override={r['override_fraction']} "
              f"reject={r['reject_fraction']} ({r['seconds']}s, {r['games_per_hour']} g/h)",
              flush=True)
        results_sorted = sorted(results, key=lambda x: -x["block_mean_placement"])
        out = dict(meta=meta, n_cells_done=len(results), cells=results,
                   ranked=results_sorted,
                   best=results_sorted[0] if results_sorted else None,
                   any_clears_2p5=any(x["clears_2p5"] for x in results))
        json.dump(out, open(a.out, "w"), indent=1)
    print("\nSWEEP DONE. Ranked:")
    for r in sorted(results, key=lambda x: -x["block_mean_placement"]):
        print(f"  belief={r['belief']} leaf={r['leaf']} K={r['K']} N={r['N']}: "
              f"{r['block_mean_placement']} CI{r['ci95']} clears2.5={r['clears_2p5']}")


if __name__ == "__main__":
    main()
