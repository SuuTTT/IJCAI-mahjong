"""cook_seq_par.py — parallel version of cook_seq (per-Match chunks, same order as cook_parallel
so alignment to cooked_single is preserved). Reuses cook_seq.process on each chunk."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
from cook_seq import process, L
HERE = os.path.dirname(os.path.abspath(__file__)); DDIR = os.path.join(HERE, "data")
DATA = os.path.join(DDIR, "data.txt")

def _worker(chunk):
    try: return process(chunk, 0)
    except Exception as e:
        sys.stderr.write("werr %s\n" % e); return np.zeros((0, L), np.int16), np.zeros((0,), np.int16)

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=64); ap.add_argument("--out", required=True)
    a = ap.parse_args(); t0 = time.time()
    with open(DATA, encoding="UTF-8") as f: lines = f.readlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("Match")]; nm = len(starts)
    per = max(1, (nm + a.workers * 2 - 1) // (a.workers * 2)); chunks = []
    for c in range(0, nm, per):
        b = starts[c + per] if c + per < nm else len(lines); chunks.append(lines[starts[c]:b])
    print(f"matches {nm} chunks {len(chunks)} ({time.time()-t0:.0f}s)", flush=True)
    with mp.Pool(a.workers) as p: res = p.map(_worker, chunks)
    seq = np.concatenate([r[0] for r in res], 0); act = np.concatenate([r[1] for r in res], 0)
    print(f"samples {len(act):,} seq{seq.shape} ({time.time()-t0:.0f}s)", flush=True)
    ca = np.load(os.path.join(DDIR, "cooked_act.npy")); n = len(act)
    print(f"ALIGN = {(len(ca)==n) and np.array_equal(act, ca.astype(np.int16))} (myN={n} cookedN={len(ca)})", flush=True)
    np.savez_compressed(a.out, seq=seq, act=act); print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__": main()
