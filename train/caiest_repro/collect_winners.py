"""
collect_winners.py — INCREMENTAL distillation-data collector for top-player games.
Drop new Botzone log batches anywhere under --root; this scans for *_full_log.json, and PER GAME:
  - if all 4 players are the SAME bot (4-same self-play)  -> extract ALL 4 seats' discard decisions
  - otherwise (mixed top bots)                            -> extract ONLY the WINNER seat's decisions
Dedups by match-id (filename prefix) via a manifest, so re-running only adds NEW games. Maintains one
cumulative npz you re-distill on:  python3 distill.py finetune_frac --champ <cumulative.npz> ...

  python3 collect_winners.py --root others/top4-players --out data/topwinners.npz
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from feature import FeatureAgent

def _meta_names(full_log_path):
    mid = os.path.basename(full_log_path).split('_')[0]
    mp = os.path.join(os.path.dirname(full_log_path), mid + '_metadata.json')
    if os.path.exists(mp):
        try:
            d = json.load(open(mp)); return mid, [p.get('name', '?') for p in d.get('players', [])]
        except Exception:
            pass
    return mid, None

def _winner(d):
    for rec in d:
        disp = (rec.get('output') or {}).get('display') or {}
        if 'score' in disp:
            sc = disp['score']; return max(range(4), key=lambda i: sc[i]) if max(sc) > 0 else None
    return None

def _extract_game(d, seats):
    """Replay one game, return (obs,mask,act) for the given seats' discard decisions."""
    O, M, A = [], [], []; quan = 0; ag = None; pend = {}
    for rec in d:
        disp = (rec.get('output') or {}).get('display') or {}
        a = disp.get('action')
        try:
            if a == 'INIT': quan = disp.get('quan', 0)
            elif a == 'DEAL':
                ag = [FeatureAgent(s) for s in range(4)]
                for s in range(4):
                    ag[s].request2obs('Wind %d' % quan); ag[s].request2obs('Deal ' + ' '.join(disp['hand'][s]))
            elif a == 'DRAW':
                p = disp['player']; t = disp['tile']; my = None
                for s in range(4):
                    r = ag[s].request2obs('Draw %s' % t) if s == p else ag[s].request2obs('Player %d Draw' % p)
                    if s == p: my = r
                if my is not None and int(my['action_mask'].sum()) > 1: pend[p] = my
            elif a == 'PLAY':
                p = disp['player']; t = disp['tile']
                if p in pend:
                    o = pend.pop(p)
                    if p in seats:
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
        except Exception:
            break
    return O, M, A

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    manifest = a.out + '.manifest'
    done = set(open(manifest).read().split()) if os.path.exists(manifest) else set()
    if os.path.exists(a.out):
        z = np.load(a.out); O = [z['obs']]; M = [z['mask']]; A = [z['act']]; prev = len(z['act'])
    else:
        O = []; M = []; A = []; prev = 0
    logs = sorted(glob.glob(os.path.join(a.root, '**', '*_full_log.json'), recursive=True) +
                  glob.glob(os.path.join(a.root, '*_full_log.json')))
    new_games = 0; mode_counts = {'same': 0, 'mixed': 0}
    for path in logs:
        mid, names = _meta_names(path)
        if mid in done or os.path.getsize(path) == 0: continue
        try: d = json.load(open(path))
        except Exception: continue
        if names and len(set(names)) == 1:
            seats = {0, 1, 2, 3}; mode_counts['same'] += 1
        else:
            w = _winner(d)
            if w is None: done.add(mid); continue
            seats = {w}; mode_counts['mixed'] += 1
        o, m, ac = _extract_game(d, seats)
        if ac:
            O.append(np.stack(o).reshape(-1, 38, 4, 9).astype(np.int8))
            M.append(np.stack(m)); A.append(np.array(ac, np.int16))
        done.add(mid); new_games += 1
    if O:
        obs = np.concatenate(O); mask = np.concatenate(M); act = np.concatenate(A)
        np.savez_compressed(a.out, obs=obs, mask=mask, act=act)
        open(manifest, 'w').write('\n'.join(sorted(done)))
        print(f"+{new_games} new games ({mode_counts['same']} 4-same all-seats, {mode_counts['mixed']} mixed winner-only)")
        print(f"cumulative: {prev} -> {len(act)} decisions  -> {a.out}")
    else:
        print("no new games found.")

if __name__ == '__main__':
    main()
