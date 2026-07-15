"""PROBE 1 — input-ablation diagnostic for big256x40_s0 (ResBNCNN 256x40, 38-plane).
Measure held-out val top-1 action accuracy; then zero each obs feature group and re-measure.
Groups: winds 0-1, hand 2-5, discards 6-21, melds 22-37.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from models_explore import ResBNCNN

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = 'data/cooked_single.npz'
CKPT = 'ckpt/big256x40_s0.pkl'
OUT = '/root/FEATURE_PROBE.txt'

GROUPS = {
    'winds(0-1)':   list(range(0, 2)),
    'hand(2-5)':    list(range(2, 6)),
    'discards(6-21)': list(range(6, 22)),
    'melds(22-37)': list(range(22, 38)),
}

def main():
    d = np.load(DATA)
    obs, mask, act = d['obs'], d['mask'], d['act'].astype(np.int64)
    n = len(act)
    # SAME val split as bc_any.py: RandomState(0).permutation, first min(20000, n//10)
    rng = np.random.RandomState(0); perm = rng.permutation(n)
    nval = min(20000, n // 10); vi = perm[:nval]
    print(f'n={n} val={nval}', flush=True)

    m = ResBNCNN(channels=256, blocks=40).to(DEV)
    sd = torch.load(CKPT, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd and not any(k.startswith(('stem','body','foot')) for k in sd):
        sd = sd['state_dict']
    m.load_state_dict(sd); m.eval()

    Ov = torch.from_numpy(obs[vi]).float()           # (nval,38,4,9)
    Mv = torch.from_numpy(mask[vi]).float()
    Av = torch.from_numpy(act[vi])

    @torch.no_grad()
    def acc(zero_planes=None):
        c = 0
        for i in range(0, nval, 8192):
            o = Ov[i:i+8192].clone()
            if zero_planes is not None:
                o[:, zero_planes, :, :] = 0
            mk = Mv[i:i+8192].to(DEV); o = o.to(DEV)
            pr = m({'is_training': False, 'obs': {'observation': o, 'action_mask': mk}}).argmax(1)
            c += (pr.cpu() == Av[i:i+8192]).sum().item()
        return c / nval

    t0 = time.time()
    base = acc(None)
    lines = [f'PROBE 1 — input ablation  ckpt={CKPT}  val_n={nval}',
             f'baseline val top-1 acc = {base:.4f}', '']
    print(lines[1], flush=True)
    res = {'baseline': round(base, 4)}
    for name, planes in GROUPS.items():
        a = acc(planes)
        drop = base - a
        ln = f'zero {name:16s}: acc={a:.4f}  drop={drop:+.4f}'
        lines.append(ln); print(ln, flush=True)
        res[name] = {'acc': round(a, 4), 'drop': round(drop, 4)}
    lines.append(''); lines.append(f'elapsed {time.time()-t0:.0f}s')
    with open(OUT, 'w') as f: f.write('\n'.join(lines) + '\n')
    print('wrote', OUT, flush=True)
    print('JSON', json.dumps(res), flush=True)

if __name__ == '__main__':
    main()
