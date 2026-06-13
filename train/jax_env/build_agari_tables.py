"""Build per-group feasibility tables for agari_jax (run once). Suit: 5^9 keys (runs allowed);
honor: 5^7 keys (no runs). Each entry = uint16 bitmask over bit(sets*2+pair), sets 0..4, pair 0..1."""
import numpy as np, itertools, sys
from functools import lru_cache
sys.setrecursionlimit(10000)

@lru_cache(maxsize=None)
def feas(c, runs):
    c = list(c); i = next((k for k, v in enumerate(c) if v > 0), None)
    if i is None: return frozenset({(0, 0)})
    out = set()
    if c[i] >= 3:
        c[i] -= 3
        for s, p in feas(tuple(c), runs): out.add((s + 1, p))
        c[i] += 3
    if runs and i + 2 < len(c) and c[i] and c[i + 1] and c[i + 2]:
        c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
        for s, p in feas(tuple(c), runs): out.add((s + 1, p))
        c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
    if c[i] >= 2:
        c[i] -= 2
        for s, p in feas(tuple(c), runs):
            if p == 0: out.add((s, 1))
        c[i] += 2
    return frozenset(out)

def build(nslots, runs, fname):
    table = np.zeros(5 ** nslots, np.uint16)
    for vec in itertools.product(range(5), repeat=nslots):
        if sum(vec) > 14: continue
        key = sum(v * (5 ** j) for j, v in enumerate(vec))
        m = 0
        for s, p in feas(vec, runs):
            if s <= 4 and p <= 1: m |= (1 << (s * 2 + p))
        table[key] = m
    np.save(fname, table); print(fname, table.nbytes // 1024, 'KB')

if __name__ == '__main__':
    import os
    d = os.path.dirname(os.path.abspath(__file__))
    build(7, False, os.path.join(d, 'agari_honor.npy'))
    build(9, True, os.path.join(d, 'agari_suit.npy'))
