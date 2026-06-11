"""
ladder_report.py — real-ladder results for OUR bots from the hourly-collected games.

Mines every collected game (future_hourly / backfill_24h / bulk_ranked_matches) for seats held by
our account's bots, and reports per-bot: games, net duplicate score, wins, deal-ins (we discarded
the winning tile = paid the 8+fan extra), draws. THE ground-truth read vs the real field — run it
after launching tables (collector picks games up within the hour).

  python3 tools/ladder_report.py                  # default: match 'wangyongyi' or 'caiest'
  python3 tools/ladder_report.py lad_chunjiandu   # match a specific deployed bot name
"""
import json, glob, os, sys, collections

ROOTS = ['others/ladder_top30_score1216/future_hourly',
         'others/ladder_top30_score1216/backfill_24h',
         'others/ladder_top30_score1216/bulk_ranked_matches']
PATTERNS = [a.lower() for a in (sys.argv[1:] or ['wangyongyi', 'caiest'])]


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = collections.defaultdict(lambda: dict(g=0, score=0, wins=0, deals=0, draws=0,
                                               opp=collections.Counter()))
    for r in ROOTS:
        for m in glob.glob(os.path.join(base, r, '**', '*_metadata.json'), recursive=True):
            try:
                md = json.load(open(m))
                names = [p.get('name', '?') for p in md.get('players', [])]
            except Exception:
                continue
            seats = [i for i, n in enumerate(names) if any(p in n.lower() for p in PATTERNS)]
            if not seats:
                continue
            mid = os.path.basename(m).replace('_metadata.json', '')
            cand = glob.glob(os.path.dirname(m) + '/../**/' + mid + '_full_log.json', recursive=True)
            if not cand:
                continue
            try:
                d = json.load(open(cand[0]))
            except Exception:
                continue
            sc = None; act = None; winner = None; last_play = None
            for rec in d:
                disp = (rec.get('output') or {}).get('display') or {}
                if disp.get('action') == 'PLAY':
                    last_play = disp.get('player')
                if isinstance(disp.get('score'), list) and len(disp['score']) == 4:
                    sc = disp['score']; act = disp.get('action'); winner = disp.get('player')
            if sc is None:
                sc, act = [0, 0, 0, 0], 'DRAW'
            for s in seats:
                R = res[names[s]]
                R['g'] += 1; R['score'] += sc[s]
                if act == 'HU' and winner == s:
                    R['wins'] += 1
                elif act == 'DRAW' or max(sc) == 0:
                    R['draws'] += 1
                elif act == 'HU' and last_play == s and winner != s:
                    R['deals'] += 1
                for i, n in enumerate(names):
                    if i not in seats:
                        R['opp'][n] += 1
    if not res:
        print('no games found for patterns:', PATTERNS); return
    for name, R in sorted(res.items(), key=lambda x: -x[1]['g']):
        g = R['g']
        print(f"\n{name}: {g} games | net {R['score']:+d} ({R['score']/g:+.2f}/g) | "
              f"wins {R['wins']} ({100*R['wins']/g:.0f}%) | deal-ins {R['deals']} ({100*R['deals']/g:.0f}%) | "
              f"draws {R['draws']} ({100*R['draws']/g:.0f}%)")
        for n, c in R['opp'].most_common(8):
            print(f"    {c:4d}  vs {n}")


if __name__ == '__main__':
    main()
