"""
curriculum_states.py — build shanten-bucketed initial hands for curriculum RL (PKU thesis §3.2).
Replay WINNING games, capture the WINNER's concealed 13-tile hand at each of their turns (only
fully-concealed, no-meld states, so shanten is well-defined), bucket by shanten distance k=0..3.
These near-win hands seed the curriculum: stage k starts the learner from a (≤k)-shanten hand,
so reward is dense early (a 0-shanten start wins ~85% per the thesis), then difficulty increases.

  python3 curriculum_states.py <out.pkl> <full_log_glob...>
"""
import os, sys, json, glob, pickle, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature import FeatureAgent
from MahjongGB import RegularShanten, SevenPairsShanten, ThirteenOrphansShanten

def _shanten(concealed):
    best = 99
    for fn in (RegularShanten, SevenPairsShanten, ThirteenOrphansShanten):
        try:
            s, _ = fn(tuple(concealed))
            if s < best: best = s
        except Exception:
            pass
    return best

def _winner(d):
    for rec in d:
        disp = (rec.get('output') or {}).get('display') or {}
        if 'score' in disp:
            sc = disp['score']
            return max(range(4), key=lambda i: sc[i]) if max(sc) > 0 else None
    return None

def build(out, globs, kmax=3):
    logs = []
    for g in globs: logs += glob.glob(g)
    buckets = collections.defaultdict(list); seen = set()
    n_games = 0
    for path in logs:
        if os.path.getsize(path) == 0: continue
        try:
            d = json.load(open(path))
        except Exception:
            continue
        w = _winner(d)
        if w is None: continue
        n_games += 1
        quan = 0; ag = None
        for rec in d:
            disp = (rec.get('output') or {}).get('display') or {}
            a = disp.get('action')
            try:
                if a == 'INIT':
                    quan = disp.get('quan', 0)
                elif a == 'DEAL':
                    ag = [FeatureAgent(s) for s in range(4)]
                    for s in range(4):
                        ag[s].request2obs('Wind %d' % quan)
                        ag[s].request2obs('Deal ' + ' '.join(disp['hand'][s]))
                elif a == 'DRAW':
                    p = disp['player']; t = disp['tile']
                    # BEFORE the winner draws: snapshot their concealed 13-tile no-meld hand
                    if p == w and ag is not None and len(ag[w].packs[0]) == 0 and len(ag[w].hand) == 13:
                        hand = sorted(ag[w].hand)
                        key = tuple(hand)
                        if key not in seen:
                            seen.add(key)
                            k = _shanten(hand)
                            if 0 <= k <= kmax:
                                buckets[k].append(hand)
                    for s in range(4):
                        ag[s].request2obs('Draw %s' % t if s == p else 'Player %d Draw' % p)
                elif a == 'PLAY':
                    p = disp['player']; t = disp['tile']
                    for s in range(4): ag[s].request2obs('Player %d Play %s' % (p, t))
                elif a == 'CHI':
                    p = disp['player']; mid = disp.get('tileCHI') or disp.get('tile')
                    for s in range(4): ag[s].request2obs('Player %d Chi %s' % (p, mid))
                elif a == 'PENG':
                    for s in range(4): ag[s].request2obs('Player %d Peng' % disp['player'])
                elif a == 'GANG':
                    for s in range(4): ag[s].request2obs('Player %d Gang' % disp['player'])
            except Exception:
                break
    pickle.dump(dict(buckets), open(out, 'wb'))
    print(f"games={n_games} | bucketed concealed hands by shanten:")
    for k in sorted(buckets): print(f"  {k}-shanten: {len(buckets[k])} hands")
    print(f"-> {out}")

if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2:])
