"""
cache_adv.py — Use the VALUE model (value_256x40, ValueMT) as a CRITIC to compute a
baseline-subtracted advantage for every decision in cooked_value.npz.

special points = placement points by rank: place 1/2/3/4 -> 4/3/2/1 points.
V_place(s) = E_softmax[place_head] . [4,3,2,1]  (critic's expected special-points for state s)
A(s) = special_points(realized deal_place) - V_place(s)

The baseline subtraction (V_place) is the KEY difference vs the prior null raw-outcome AWR:
A is centred per-state by the critic's expectation, so it isolates the surprise of this seat's
outcome given the situation -- a real learning signal IFF the critic has signal.

Writes data/adv_cache.npz (compressed): adv (float32 N,), vplace (float32 N,), realized_pts (int8 N,).
Aligned 1:1 with cooked_value rows (same order).
"""
import os, sys, time, argparse
sys.path.insert(0, '/root/IJCAI-mahjong/train/caiest_repro')
import numpy as np, torch
from train_value import ValueMT

DATA = '/root/IJCAI-mahjong/train/caiest_repro/data/cooked_value.npz'
VAL = '/root/IJCAI-mahjong/train/caiest_repro/ckpt/value_256x40.pkl'
OUT = '/root/IJCAI-mahjong/train/caiest_repro/data/adv_cache.npz'
PTS = np.array([4., 3., 2., 1.], dtype=np.float64)  # place 1..4 -> special points

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--val', default=VAL)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--bs', type=int, default=8192)
    a = ap.parse_args()
    dev = f'cuda:{a.gpu}'; torch.cuda.set_device(a.gpu)
    ck = torch.load(a.val, map_location='cpu')
    net = ValueMT(ck['channels'], ck['blocks']).to(dev)
    net.load_state_dict(ck['state']); net.eval()
    print(f'critic ch={ck["channels"]} blocks={ck["blocks"]} loaded', flush=True)

    z = np.load(a.data)
    obs = z['obs']; place = z['deal_place'].astype(np.int64)  # 1..4
    N = len(place)
    realized_pts = (5 - place).astype(np.float64)             # 1->4 .. 4->1
    vplace = np.empty(N, dtype=np.float64)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, a.bs):
            ob = torch.from_numpy(obs[i:i+a.bs].astype(np.float32)).to(dev)
            pl, _, _ = net(ob)
            sm = torch.softmax(pl, dim=1).cpu().numpy()        # (b,4) over place 1..4
            vplace[i:i+len(sm)] = sm @ PTS
            if (i // a.bs) % 50 == 0:
                print(f'  {i}/{N} ({time.time()-t0:.0f}s)', flush=True)
    adv = realized_pts - vplace
    # report
    print(f'N={N}', flush=True)
    print(f'realized_pts: mean={realized_pts.mean():.4f} (uniform place -> expect 2.5)', flush=True)
    print(f'V_place: mean={vplace.mean():.4f} std={vplace.std():.4f} min={vplace.min():.4f} max={vplace.max():.4f}', flush=True)
    print(f'ADV: mean={adv.mean():+.4f} std={adv.std():.4f} min={adv.min():+.3f} max={adv.max():+.3f}', flush=True)
    qs = np.percentile(adv, [1,5,25,50,75,95,99])
    print('ADV pctiles [1,5,25,50,75,95,99]: ' + ' '.join(f'{q:+.3f}' for q in qs), flush=True)
    # is the critic informative? correlation of V_place with realized outcome
    r = np.corrcoef(vplace, realized_pts)[0,1]
    print(f'corr(V_place, realized_pts) = {r:+.4f}  (critic predictiveness on this data)', flush=True)
    # fraction of |A| that is "small" -- degenerate check
    print(f'frac |adv|<0.25 = {(np.abs(adv)<0.25).mean():.3f}   frac |adv|>1.0 = {(np.abs(adv)>1.0).mean():.3f}', flush=True)
    np.savez_compressed(a.out, adv=adv.astype(np.float32), vplace=vplace.astype(np.float32),
                        realized_pts=realized_pts.astype(np.int8))
    print(f'wrote {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)', flush=True)

if __name__ == '__main__':
    main()
