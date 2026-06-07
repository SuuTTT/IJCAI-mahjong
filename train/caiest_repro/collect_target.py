"""
collect_target.py — like collect_winners.py but extracts the TARGET player's seat (from
*_metadata.json targetPlayer, e.g. the rank-1 bot in tournament logs) in EVERY game, PLUS the
winner's seat when someone else won (diverse top-bot teacher). BC on a strong policy's decisions
is valid regardless of outcome — distill100b itself was trained on all 4 self-play seats.

  python3 collect_target.py --root <logs> --out data/topwinners_plus.npz
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collect_winners import _winner, _extract_game

def _meta(full_log_path):
    mid = os.path.basename(full_log_path).split('_')[0]
    mp = os.path.join(os.path.dirname(full_log_path), mid + '_metadata.json')
    if os.path.exists(mp):
        try:
            d = json.load(open(mp))
            names = [p.get('name', '?') for p in d.get('players', [])]
            return mid, names, d.get('targetPlayer')
        except Exception:
            pass
    return mid, None, None

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
    new_games = 0; n_target = 0; n_winner_extra = 0
    for path in logs:
        mid, names, target = _meta(path)
        if mid in done or os.path.getsize(path) == 0: continue
        try: d = json.load(open(path))
        except Exception: continue
        seats = set()
        if names and target and target in names:
            seats.add(names.index(target)); n_target += 1
        w = _winner(d)
        if w is not None and w not in seats:
            seats.add(w); n_winner_extra += 1
        if not seats:
            done.add(mid); continue
        o, m, ac = _extract_game(d, seats)
        if ac:
            O.append(np.stack(o).reshape(-1, 38, 4, 9).astype(np.int8))
            M.append(np.stack(m)); A.append(np.array(ac, np.int16))
        done.add(mid); new_games += 1
    if O:
        obs = np.concatenate(O); mask = np.concatenate(M); act = np.concatenate(A)
        np.savez_compressed(a.out, obs=obs, mask=mask, act=act)
        open(manifest, 'w').write('\n'.join(sorted(done)))
        print(f"+{new_games} games (target-seat in {n_target}, extra winner-seat in {n_winner_extra})")
        print(f"cumulative: {prev} -> {len(act)} decisions  -> {a.out}")
    else:
        print("no new games found.")

if __name__ == '__main__':
    main()
