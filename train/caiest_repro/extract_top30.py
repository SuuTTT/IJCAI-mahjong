"""
extract_top30.py — extract TOP-30 players' discard decisions from collected ladder games
(others/ladder_top30_score1216/...). For each game, a seat is a TOP-30 teacher iff its
"[author]botName" is in the ranking snapshot's top-30. Extracts those seats (all4 -> 4 seats,
exactly3 -> 3 seats). Real strong-ladder distribution = the best on-distribution teacher we have.

  python3 extract_top30.py --root <dir> --ranking ranking_snapshot.json --out data/top30.npz
  python3 extract_top30.py --root <dir> --ranking ... --player chunjiandu --out data/chunjiandu.npz  # coherent single teacher
  python3 extract_top30.py --root <dir> --ranking ... --profile                                       # count decisions per top player
"""
import os, sys, json, glob, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
# NOTE: _extract_game (-> feature -> MahjongGB) imported lazily inside main(), so --profile
# runs anywhere (no MahjongGB needed); only the actual extraction needs it (run on a box).

def top30_names(ranking_path):
    d = json.load(open(ranking_path))
    return set('[%s]%s' % (e.get('author'), e.get('botName')) for e in d if e.get('rank', 99) <= 30)

def _meta(path):
    mid = os.path.basename(path).split('_')[0]
    for mp in (os.path.join(os.path.dirname(path), mid + '_metadata.json'),
               os.path.join(os.path.dirname(os.path.dirname(path)), 'metadata', mid + '_metadata.json')):
        if os.path.exists(mp):
            try: return mid, [p.get('name', '?') for p in json.load(open(mp)).get('players', [])]
            except Exception: pass
    return mid, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--ranking', required=True)
    ap.add_argument('--out'); ap.add_argument('--player', default=''); ap.add_argument('--profile', action='store_true')
    ap.add_argument('--since', default='', help='YYYY-MM-DD: skip games older than this (decode Botzone ObjectId ts) -> filter out stale bot versions')
    a = ap.parse_args()
    since_ts = 0
    if a.since:
        import datetime
        since_ts = int(datetime.datetime.strptime(a.since, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    top = top30_names(a.ranking)
    logs = sorted(glob.glob(os.path.join(a.root, '**', '*full_log*.json'), recursive=True))
    if a.profile:
        c = collections.Counter()
        for path in logs:
            _, names = _meta(path)
            if names:
                for n in set(names):
                    if n in top: c[n] += 1
        for n, k in c.most_common(): print(f"{k:5d}  {n}")
        print(f"\n{len(c)} top-30 players over {len(logs)} games")
        return
    from collect_winners import _extract_game        # lazy (needs feature -> MahjongGB)
    O, M, A = [], [], []; seen = set(); ng = 0; nseat = 0
    skipped_old = 0
    for path in logs:
        mid, names = _meta(path)
        if not names or mid in seen or os.path.getsize(path) == 0: continue
        if since_ts:                                   # filter stale bot versions by game date (ObjectId ts)
            try:
                if int(mid[:8], 16) < since_ts: skipped_old += 1; continue
            except Exception: pass
        if a.player:
            pls = [p for p in a.player.split(',') if p]    # comma-list -> match ANY (focused multi-teacher)
            seats = {i for i, n in enumerate(names) if any(p in n for p in pls)}
        else:
            seats = {i for i, n in enumerate(names) if n in top}
        if not seats: continue
        try: d = json.load(open(path))
        except Exception: continue
        o, m, ac = _extract_game(d, seats)
        if ac:
            O.append(np.stack(o).reshape(-1, 38, 4, 9).astype(np.int8)); M.append(np.stack(m)); A.append(np.array(ac, np.int16))
            nseat += len(seats)
        seen.add(mid); ng += 1
    if not O: print("no decisions extracted"); return
    obs = np.concatenate(O); mask = np.concatenate(M); act = np.concatenate(A)
    np.savez_compressed(a.out, obs=obs, mask=mask, act=act)
    print(f"{ng} games, {nseat} top-30 seats -> {len(act)} decisions -> {a.out}" + (f" (skipped {skipped_old} pre-{a.since})" if since_ts else ""))

if __name__ == '__main__':
    main()
