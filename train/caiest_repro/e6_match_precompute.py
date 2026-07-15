"""e6_match_precompute.py — E6-P3 eval hands, precomputed under BOTH seat-0 modes.

All policies are deterministic and hands are independent given (field, seed), so
every eval hand is played ONCE with kd (kdens3) and ONCE with aug (aug_s0) at
seat 0; e6_match_assemble.py then builds every arm (always-kd / always-aug /
oracle / switcher) from the same outcomes — exact seed pairing across arms.

Seeds: fresh 30M+ block, disjoint from all prior E6 blocks.

  python3 e6_match_precompute.py --nmatches 2000 --hands 8 --workers 80 \
      --seed0 30000000 --out data/e6_match_eval.npz
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
from e6_match_common import (FIELD_ORDER, play_hand, rec_to_row, NROW,
                             match_seed, preload_all)

MODES = ("kd", "aug")


def _work(arg):
    fi, field, mi, mode, seed0, M = arg
    rows = [rec_to_row(play_hand(field, match_seed(seed0, fi, mi, h), mode))
            for h in range(M)]
    return fi, mi, mode, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmatches", type=int, default=2000)
    ap.add_argument("--hands", type=int, default=8)
    ap.add_argument("--workers", type=int, default=80)
    ap.add_argument("--seed0", type=int, default=30000000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    preload_all()
    N, M = a.nmatches, a.hands
    X = {m: np.zeros((4, N, M, NROW), dtype=np.float32) for m in MODES}
    for fi, field in enumerate(FIELD_ORDER):
        t0 = time.time()
        args = [(fi, field, mi, mode, a.seed0, M)
                for mi in range(N) for mode in MODES]
        done = 0
        with mp.Pool(a.workers) as p:
            for fi_, mi, mode, rows in p.imap_unordered(_work, args, chunksize=2):
                X[mode][fi_, mi] = np.asarray(rows, dtype=np.float32)
                done += 1
                if done % 1000 == 0:
                    print(f"  {field}: {done}/{len(args)} match-arms "
                          f"({time.time()-t0:.1f}s)", flush=True)
        print(f"FIELD {field}: {N} matches x {M} hands x 2 modes "
              f"({time.time()-t0:.1f}s)", flush=True)
        np.savez_compressed(a.out + f".part{fi}",
                            **{f"X_{m}": X[m][fi] for m in MODES})
    np.savez_compressed(a.out, X_kd=X["kd"], X_aug=X["aug"], seed0=a.seed0,
                        fields=np.array(FIELD_ORDER), hands=M)
    for fi in range(4):
        pp = a.out + f".part{fi}.npz"
        if os.path.exists(pp):
            os.remove(pp)
    print("SAVED", a.out, X["kd"].shape, flush=True)


if __name__ == "__main__":
    main()
