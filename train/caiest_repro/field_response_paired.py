"""PAIRED field-response replication.

P1 flaw (found 2026-07-13): each (policy,field) cell used a DISJOINT seed block
-> cross-policy diffs were UNPAIRED across different wall sets (audit-class error).
Here: per field, all policies play the SAME seed list; diffs are per-seed PAIRED.
"""
import argparse, json, math, multiprocessing as mp, os, time
from field_response import POLICIES, FIELDS, POL_ORDER, FIELD_ORDER, _work, _stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngames", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=9000000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cells, paired = {}, {}
    meta = dict(design="PAIRED: per field, identical seed list for every policy; "
                       "diffs are per-seed paired (same walls)",
                ngames_per_cell=a.ngames, seed0=a.seed0, started=time.strftime("%F %T"))
    raw = {}  # (pol,field) -> list of (rank,score,...) aligned by seed index
    for fi, field in enumerate(FIELD_ORDER):
        s0 = a.seed0 + fi * 100000  # block per FIELD, shared across policies
        for pol in POL_ORDER:
            t0 = time.time()
            args = [(s0 + i, POLICIES[pol], FIELDS[field]) for i in range(a.ngames)]
            with mp.Pool(a.workers) as p:
                res = p.map(_work, args, chunksize=4)
            raw[(pol, field)] = res
            ranks = [r[0] for r in res]; scores = [r[1] for r in res]
            cells[f"{pol}|{field}"] = dict(policy=pol, field=field, n=len(res), seed0=s0,
                rank=_stats(ranks), score=_stats(scores),
                dealin_rate=round(sum(r[2] for r in res) / len(res), 4),
                win_rate=round(sum(r[3] for r in res) / len(res), 4),
                seconds=round(time.time() - t0, 1))
            _dump(a.out, meta, cells, paired)
    # paired diffs
    for A, B in [(x, y) for i, x in enumerate(POL_ORDER) for y in POL_ORDER[i+1:]]:
        pr = {}
        for metric, idx in (("rank", 0), ("score", 1)):
            per_field = {}
            for field in FIELD_ORDER:
                ra, rb = raw.get((A, field)), raw.get((B, field))
                if not ra or not rb:
                    continue
                ds = [x[idx] - y[idx] for x, y in zip(ra, rb)]
                m = sum(ds) / len(ds)
                se = (sum((d - m) ** 2 for d in ds) / (len(ds) - 1)) ** 0.5 / len(ds) ** 0.5
                per_field[field] = dict(diff=round(m, 4),
                    ci95=[round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)],
                    significant=bool(abs(m) > 1.96 * se))
            pr[metric] = per_field
        paired[f"{A} vs {B}"] = pr
    _dump(a.out, meta, cells, paired, done=True)
    print("PAIRED_DONE", json.dumps(paired.get("kdens3 vs aug_s0", {}))[:600])


def _dump(path, meta, cells, paired, done=False):
    with open(path, "w") as f:
        json.dump(dict(meta=meta, cells=cells, paired_diffs=paired, done=done), f, indent=1)


if __name__ == "__main__":
    main()
