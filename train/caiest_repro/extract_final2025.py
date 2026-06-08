"""
extract_final2025.py — single-pass extractor for ijcai2025 final logs. For each game, replays once
and writes EVERY listed agent's discard decisions to a per-agent npz (accumulating across games),
plus records each seat's final score so we can rank the field / pick the champion.

  python3 extract_final2025.py --root <dir> --outdir data/agents2025 [--names "A" "B" ...] [--min 800]

Without --names, extracts ALL agents seen. --min drops agents with fewer than N games (after the pass).
Writes <outdir>/<safe_name>.npz and <outdir>/_scores.json (name -> [n_games, total_score, mean]).
"""
import os, sys, json, glob, argparse, collections, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from feature import FeatureAgent

def _safe(name):
    return hashlib.md5(name.encode()).hexdigest()[:10]

def _names(path):
    mid = os.path.basename(path).split('_')[0]
    mp = os.path.join(os.path.dirname(path), mid + '_metadata.json')
    try:
        return mid, [p.get('name', '?') for p in json.load(open(mp)).get('players', [])]
    except Exception:
        return mid, None

def _replay(d, want_seats):
    """Replay one game; return {seat: (O,M,A)} for seats in want_seats, plus final scores list."""
    quan = 0; ag = None; pend = {}; out = {s: ([], [], []) for s in want_seats}; scores = None
    try:
        for rec in d:
            disp = (rec.get('output') or {}).get('display') or {}
            a = disp.get('action')
            if not a: continue
            if a == 'INIT': quan = disp.get('quan', 0)
            elif a == 'DEAL':
                ag = [FeatureAgent(s) for s in range(4)]
                for s in range(4):
                    ag[s].request2obs('Wind %d' % quan); ag[s].request2obs('Deal ' + ' '.join(disp['hand'][s]))
            elif a == 'DRAW':
                p = disp['player']; t = disp['tile']
                for s in range(4):
                    r = ag[s].request2obs('Draw %s' % t) if s == p else ag[s].request2obs('Player %d Draw' % p)
                    if s == p: pend[p] = r
            elif a == 'PLAY':
                p = disp['player']; t = disp['tile']
                if p in pend:
                    o = pend.pop(p)
                    if p in want_seats and int(o['action_mask'].sum()) > 1:
                        O, M, A = out[p]
                        O.append(o['observation'].astype(np.int8)); M.append(o['action_mask'].astype(np.bool_))
                        A.append(ag[p].OFFSET_ACT['Play'] + ag[p].OFFSET_TILE[t])
                for s in range(4): ag[s].request2obs('Player %d Play %s' % (p, t))
            elif a == 'CHI':
                p = disp['player']; mid = disp.get('tileCHI') or disp.get('tile')
                for s in range(4): ag[s].request2obs('Player %d Chi %s' % (p, mid))
            elif a == 'PENG':
                for s in range(4): ag[s].request2obs('Player %d Peng' % disp['player'])
            elif a == 'GANG':
                for s in range(4): ag[s].request2obs('Player %d Gang' % disp['player'])
            if 'score' in disp: scores = disp['score']
    except Exception:
        pass
    return out, scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--outdir', required=True)
    ap.add_argument('--names', nargs='*'); ap.add_argument('--min', type=int, default=800)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    logs = sorted(glob.glob(os.path.join(a.root, '**', '*_full_log.json'), recursive=True))
    print(f"{len(logs)} games", flush=True)
    want = set(a.names) if a.names else None
    acc = collections.defaultdict(lambda: ([], [], []))         # name -> (O,M,A)
    stat = collections.defaultdict(lambda: [0, 0.0])            # name -> [games, total_score]
    for i, path in enumerate(logs):
        if i % 4000 == 0: print(f"  {i}/{len(logs)}", flush=True)
        if os.path.getsize(path) == 0: continue
        mid, names = _names(path)
        if not names: continue
        seats = {s for s, n in enumerate(names) if (want is None or n in want)}
        if not seats: continue
        try: d = json.load(open(path))
        except Exception: continue
        per, scores = _replay(d, seats)
        for s in seats:
            O, M, A = per[s]
            if A:
                aO, aM, aA = acc[names[s]]
                aO.append(np.stack(O).reshape(-1, 38, 4, 9).astype(np.int8))
                aM.append(np.stack(M)); aA.append(np.array(A, np.int16))
            stat[names[s]][0] += 1
            if scores: stat[names[s]][1] += scores[s]
    summary = {}
    for name, (O, M, A) in acc.items():
        ng = stat[name][0]
        if ng < a.min: continue
        obs = np.concatenate(O); mask = np.concatenate(M); act = np.concatenate(A)
        fn = os.path.join(a.outdir, _safe(name) + '.npz')
        np.savez_compressed(fn, obs=obs, mask=mask, act=act)
        summary[name] = {'file': os.path.basename(fn), 'games': ng,
                         'decisions': int(len(act)), 'mean_score': round(stat[name][1] / max(1, ng), 2)}
        print(f"{name}: {ng} games, {len(act)} decisions, mean {summary[name]['mean_score']:+.2f} -> {fn}", flush=True)
    json.dump(summary, open(os.path.join(a.outdir, '_scores.json'), 'w'), ensure_ascii=False, indent=1)
    print("wrote", os.path.join(a.outdir, '_scores.json'), flush=True)

if __name__ == '__main__':
    main()
