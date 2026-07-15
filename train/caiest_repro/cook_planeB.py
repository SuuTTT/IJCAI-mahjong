"""
cook_planeB.py — precompute the shanten/useful-tile planes (set B) for ALL cooked samples,
parallel across cores. Writes data/planeB.npy (N,4,4,9) float32. Idempotent-ish: writes via
a temp memmap then renames. Use --limit for a smoke test.

  python3 cook_planeB.py --workers 120
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
import featplus

HERE = os.path.dirname(os.path.abspath(__file__)); DDIR = os.path.join(HERE, 'data')
_O = None

def _init(path):
    global _O
    _O = np.load(path, mmap_mode='r')

def _chunk(arg):
    lo, hi = arg
    out = np.zeros((hi - lo, 4, 4, 9), np.float32)
    for j, i in enumerate(range(lo, hi)):
        out[j] = featplus._shanten_planes_one(np.asarray(_O[i], np.float32), None)
    return lo, hi, out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obs', default=os.path.join(DDIR, 'cooked_obs.npy'))
    ap.add_argument('--out', default=os.path.join(DDIR, 'planeB.npy'))
    ap.add_argument('--workers', type=int, default=120)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--chunk', type=int, default=4000)
    a = ap.parse_args()
    o = np.load(a.obs, mmap_mode='r'); N = len(o) if a.limit <= 0 else min(a.limit, len(o))
    print(f"cooking planeB for N={N:,} -> {a.out}", flush=True)
    res = np.lib.format.open_memmap(a.out + '.tmp', mode='w+', dtype=np.float32, shape=(N, 4, 4, 9))
    tasks = [(lo, min(lo + a.chunk, N)) for lo in range(0, N, a.chunk)]
    t0 = time.time(); done = 0
    with mp.Pool(a.workers, initializer=_init, initargs=(a.obs,)) as p:
        for lo, hi, out in p.imap_unordered(_chunk, tasks):
            res[lo:hi] = out; done += hi - lo
            if done % 100000 < a.chunk:
                el = time.time() - t0; print(f"  {done:,}/{N:,} ({done/el:.0f}/s, ETA {(N-done)/max(1,done/el)/60:.1f}m)", flush=True)
    res.flush(); del res
    os.replace(a.out + '.tmp', a.out)
    print(f"DONE planeB {N:,} in {(time.time()-t0)/60:.1f}m -> {a.out}", flush=True)

if __name__ == '__main__':
    main()
