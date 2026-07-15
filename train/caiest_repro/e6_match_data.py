"""e6_match_data.py — E6-P3 estimator training data: M-hand matches, kdens3 seat-0.

For each of the 4 Phase-1 fields, plays --nmatches matches of --hands hands with
kdens3 driving seat 0 (the pre-switch policy, i.e. exactly the distribution the
match switcher sees while still estimating). Seeds: seed0 + field_idx*1e6 +
match*16 + hand — a fresh block (default 20M+) disjoint from Phase-1 (7.0-8.2M),
the P2 estimator data (10M+), the P2 gate (8M+), and the P3 eval block (30M+).

  python3 e6_match_data.py --nmatches 2000 --hands 8 --workers 80 \
      --seed0 20000000 --out data/e6_match_train.npz
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
from e6_match_common import (FIELD_ORDER, play_hand, rec_to_row, NROW,
                             match_seed, preload_all)


def _work(arg):
    fi, field, mi, seed0, M = arg
    rows = [rec_to_row(play_hand(field, match_seed(seed0, fi, mi, h), "kd"))
            for h in range(M)]
    return fi, mi, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmatches", type=int, default=2000)
    ap.add_argument("--hands", type=int, default=8)
    ap.add_argument("--workers", type=int, default=80)
    ap.add_argument("--seed0", type=int, default=20000000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    preload_all()
    N, M = a.nmatches, a.hands
    X = np.zeros((4, N, M, NROW), dtype=np.float32)
    for fi, field in enumerate(FIELD_ORDER):
        t0 = time.time()
        args = [(fi, field, mi, a.seed0, M) for mi in range(N)]
        with mp.Pool(a.workers) as p:
            for fi_, mi, rows in p.imap_unordered(_work, args, chunksize=2):
                X[fi_, mi] = np.asarray(rows, dtype=np.float32)
        print(f"FIELD {field}: {N} matches x {M} hands "
              f"({time.time()-t0:.1f}s)", flush=True)
    np.savez_compressed(a.out, X=X, seed0=a.seed0,
                        fields=np.array(FIELD_ORDER), hands=M)
    print("SAVED", a.out, X.shape, flush=True)


if __name__ == "__main__":
    main()
