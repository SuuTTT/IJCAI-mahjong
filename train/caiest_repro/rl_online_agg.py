"""rl_online_agg.py — aggregate multi-block gate JSONs for one RL snapshot into a CI verdict
vs the aug_s0 self-play null (2.500), appended to RL_ONLINE_RESULTS.json."""
import os, sys, json, glob, math, time

RESULTS = '/root/IJCAI-mahjong/train/caiest_repro/RL_ONLINE_RESULTS.json'
NULL = 2.500  # aug_s0 vs aug_s0 calibrated placement

def main():
    tag = sys.argv[1]; blockdir = sys.argv[2]
    blocks = sorted(glob.glob(os.path.join(blockdir, f'{tag}_b*.json')))
    pts = []; games = 0; first = []; fourth = []; cand = ref = ''
    for b in blocks:
        try:
            d = json.load(open(b))
        except Exception:
            continue
        pts.append(d['placement_pts']); games += d.get('games', 0)
        first.append(d.get('first_pct', 0)); fourth.append(d.get('fourth_pct', 0))
        cand = d.get('cand', cand); ref = d.get('ref', ref)
    n = len(pts)
    if n == 0:
        print(f"[agg] no blocks for {tag}"); return
    mean = sum(pts) / n
    if n >= 2:
        sd = math.sqrt(sum((x - mean) ** 2 for x in pts) / (n - 1))
        se = sd / math.sqrt(n)
    else:
        sd = se = float('nan')
    lo = mean - 1.96 * se if n >= 2 else float('nan')
    hi = mean + 1.96 * se if n >= 2 else float('nan')
    beats = (n >= 2) and (lo > NULL)
    rec = dict(tag=tag, cand=cand, ref=ref, blocks=n, block_pts=[round(x, 4) for x in pts],
               games_total=games, placement_mean=round(mean, 4),
               placement_sd=round(sd, 4) if n >= 2 else None,
               placement_se=round(se, 4) if n >= 2 else None,
               ci95_lo=round(lo, 4) if n >= 2 else None,
               ci95_hi=round(hi, 4) if n >= 2 else None,
               null=NULL, delta_vs_null=round(mean - NULL, 4),
               ci_beats_null=bool(beats),
               first_pct=round(sum(first) / n, 2), fourth_pct=round(sum(fourth) / n, 2),
               ts=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    allrec = []
    if os.path.exists(RESULTS):
        try: allrec = json.load(open(RESULTS))
        except Exception: allrec = []
    allrec = [r for r in allrec if r.get('tag') != tag]
    allrec.append(rec)
    json.dump(allrec, open(RESULTS, 'w'), indent=2)
    verdict = 'CI-BEATS aug_s0' if beats else ('above-null(not-CI)' if mean > NULL else 'null/regress')
    print(f"[agg] {tag} mean={mean:.4f} sd={sd:.4f} ci95=[{lo:.4f},{hi:.4f}] vs {NULL} -> {verdict}", flush=True)

if __name__ == '__main__':
    main()
