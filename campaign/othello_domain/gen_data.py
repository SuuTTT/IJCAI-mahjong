"""Generate imitation data for a depth-D minimax teacher on 6x6 Othello.

Self-play the depth-D teacher against itself with random openings + epsilon-greedy
exploration for state diversity. At EVERY visited state, record the teacher's
GREEDY depth-D best move as the label (the coherent target). Labels are always
depth-D decisions; exploration only diversifies which states we visit.

Output npz: me[], opp[] (int64 bitboards, mover perspective), y[] (label square),
plus metadata. Parallel across CPU workers.

Usage: python gen_data.py --depth D --n_states N --out FILE --workers W --seed S
"""
import argparse, json, os, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np
import othello as oth
import teacher as T


def gen_worker(args):
    depth, n_target, seed, p_explore, open_plies = args
    rng = np.random.RandomState(seed)
    import random as _r
    prng = _r.Random(seed ^ 0x9E3779B9)
    me_list, opp_list, y_list = [], [], []
    while len(y_list) < n_target:
        me, opp = oth.initial()
        # random opening
        nopen = open_plies
        ply = 0
        passes = 0
        while True:
            if oth.is_terminal(me, opp):
                break
            moves = oth.legal_move_list(me, opp)
            if not moves:
                me, opp = opp, me
                passes += 1
                if passes >= 2:
                    break
                continue
            passes = 0
            if ply < nopen:
                sq = prng.choice(moves)          # random opening
            else:
                best = T.best_move(me, opp, depth)   # greedy label
                # record this state with the greedy label
                me_list.append(me)
                opp_list.append(opp)
                y_list.append(best)
                # exploration: sometimes deviate for diversity
                if prng.random() < p_explore:
                    sq = prng.choice(moves)
                else:
                    sq = best
            me, opp = oth.apply_move(me, opp, sq)
            ply += 1
            if len(y_list) >= n_target:
                break
    return (np.array(me_list[:n_target], dtype=np.int64),
            np.array(opp_list[:n_target], dtype=np.int64),
            np.array(y_list[:n_target], dtype=np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--n_states", type=int, default=120000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--p_explore", type=float, default=0.12)
    ap.add_argument("--open_plies", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    per = a.n_states // a.workers + 1
    tasks = [(a.depth, per, a.seed + 1000 * i, a.p_explore, a.open_plies)
             for i in range(a.workers)]
    with mp.Pool(a.workers) as pool:
        res = pool.map(gen_worker, tasks)
    me = np.concatenate([r[0] for r in res])[:a.n_states]
    opp = np.concatenate([r[1] for r in res])[:a.n_states]
    y = np.concatenate([r[2] for r in res])[:a.n_states]
    np.savez_compressed(a.out, me=me, opp=opp, y=y)
    meta = {"depth": a.depth, "n_states": int(len(y)), "seed": a.seed,
            "p_explore": a.p_explore, "open_plies": a.open_plies,
            "board": "6x6", "workers": a.workers,
            "gen_seconds": round(time.time() - t0, 1)}
    with open(a.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[gen] depth={a.depth} states={len(y)} "
          f"secs={meta['gen_seconds']} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
