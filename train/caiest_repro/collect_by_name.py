"""
collect_by_name.py — extract ONE named agent's discard decisions across every game it appears in
(for datasets whose metadata lists players but has no targetPlayer field, e.g. ijcai2025 final logs).
Builds a per-agent npz = a BC teacher / imitation-opponent training set.

  python3 collect_by_name.py --root <dir> --name "[Jirachi]PAMA" --out data/agent_PAMA.npz
  # profile mode: list agents + game counts in a folder
  python3 collect_by_name.py --root <dir> --profile
"""
import os, sys, json, glob, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collect_winners import _extract_game   # same replay → (obs,mask,act) for given seats

def _logs(root):
    return sorted(glob.glob(os.path.join(root, '**', '*_full_log.json'), recursive=True) +
                  glob.glob(os.path.join(root, '*_full_log.json')))

def _meta_names(full_log_path):
    mid = os.path.basename(full_log_path).split('_')[0]
    mp = os.path.join(os.path.dirname(full_log_path), mid + '_metadata.json')
    if os.path.exists(mp):
        try:
            return mid, [p.get('name', '?') for p in json.load(open(mp)).get('players', [])]
        except Exception:
            pass
    return mid, None

def profile(root):
    c = collections.Counter()
    for path in _logs(root):
        _, names = _meta_names(path)
        if names:
            for n in set(names): c[n] += 1
    for name, n in c.most_common():
        print(f"{n:6d}  {name}")
    print(f"\n{len(c)} distinct agents over {len(_logs(root))} games")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--name'); ap.add_argument('--out')
    ap.add_argument('--profile', action='store_true')
    a = ap.parse_args()
    if a.profile:
        profile(a.root); return
    assert a.name and a.out, "need --name and --out"
    manifest = a.out + '.manifest'
    done = set(open(manifest).read().split()) if os.path.exists(manifest) else set()
    if os.path.exists(a.out):
        z = np.load(a.out); O = [z['obs']]; M = [z['mask']]; A = [z['act']]; prev = len(z['act'])
    else:
        O = []; M = []; A = []; prev = 0
    ngames = 0
    for path in _logs(a.root):
        mid, names = _meta_names(path)
        if mid in done or not names or a.name not in names or os.path.getsize(path) == 0:
            continue
        try: d = json.load(open(path))
        except Exception: continue
        seat = names.index(a.name)
        o, m, ac = _extract_game(d, {seat})
        if ac:
            O.append(np.stack(o).reshape(-1, 38, 4, 9).astype(np.int8))
            M.append(np.stack(m)); A.append(np.array(ac, np.int16))
        done.add(mid); ngames += 1
    if O:
        obs = np.concatenate(O); mask = np.concatenate(M); act = np.concatenate(A)
        np.savez_compressed(a.out, obs=obs, mask=mask, act=act)
        open(manifest, 'w').write('\n'.join(sorted(done)))
        print(f"+{ngames} games for {a.name}: {prev} -> {len(act)} decisions -> {a.out}")
    else:
        print(f"no new games for {a.name}")

if __name__ == '__main__':
    main()
