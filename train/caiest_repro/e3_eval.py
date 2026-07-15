#!/usr/bin/env python3
"""E3 eval-only: load each trained ckpt, compute held-out metrics on the FIXED
RandomState(0) split (n_eval=586,581, 0% overlap asserted), write per-model JSON
matching what e3_aggregate.py expects (E3_json/*.json).

Reuses ValueMT, the split, and the metric code from train_value.py verbatim.
"""
import os, sys, json, glob, argparse
sys.path.insert(0, '/root/IJCAI-mahjong/train/caiest_repro')
import numpy as np, torch, torch.nn as nn
from train_value import ValueMT, auc   # reuse exact model + AUC from trainer

BASE = '/root/IJCAI-mahjong/train/caiest_repro'
DATA = f'{BASE}/data/cooked_value.npz'
CKDIR = f'{BASE}/ckpt/e3'
JDIR = f'{BASE}/E3_json'
os.makedirs(JDIR, exist_ok=True)

# ckpt -> (channels, blocks, seed) ; epochs trained (from campaign: 8 epochs, matches writeup)
CKPTS = [
    ('value_64x6_s0.pkl',   0),
    ('value_128x20_s0.pkl', 0),
    ('value_128x40_s0.pkl', 0),
    ('value_192x24_s0.pkl', 0),
    ('value_256x40_s0.pkl', 0),
    ('value_256x40_s1.pkl', 1),
]
EPOCHS = 8  # all E3 capacities trained 8 epochs (per e3_aggregate writeup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    dev = f'cuda:{a.gpu}'
    torch.cuda.set_device(a.gpu)

    print('loading data...', flush=True)
    z = np.load(DATA)
    obs = z['obs']
    place = z['deal_place'].astype(np.int64) - 1     # 0..3
    fourth = (z['deal_place'] == 4).astype(np.float32)
    score = (z['deal_score'].astype(np.float32)) / 48.0
    N = len(place)

    # --- IDENTICAL fixed split as trainer ---
    rng = np.random.RandomState(0); idx = rng.permutation(N)
    nval = N // 10; vi, ti = idx[:nval], idx[nval:]
    assert nval == 586581, f'n_eval mismatch: {nval}'
    assert len(np.intersect1d(vi, ti)) == 0, 'train/val overlap!'
    print(f'N={N} train={len(ti)} val={nval} (0% overlap OK)', flush=True)

    yp = place[vi]; yf = fourth[vi]; ys = score[vi]

    @torch.no_grad()
    def evaluate(net):
        net.eval(); pl_logits=[]; f_logits=[]; s_pred=[]
        for i in range(0, nval, 4096):
            b = vi[i:i+4096]
            ob = torch.from_numpy(obs[b].astype(np.float32)).to(dev)
            pl, fo, sc = net(ob)
            pl_logits.append(pl.cpu().numpy()); f_logits.append(fo.cpu().numpy()); s_pred.append(sc.cpu().numpy())
        pl = np.concatenate(pl_logits); fo = np.concatenate(f_logits); sp = np.concatenate(s_pred)
        acc = float((pl.argmax(1) == yp).mean())
        sm = np.exp(pl - pl.max(1, keepdims=True)); sm /= sm.sum(1, keepdims=True)
        exp_place = (sm * np.array([1,2,3,4])).sum(1)
        place_mae = float(np.abs(exp_place - (yp+1)).mean())
        a4 = auc(yf, fo)
        smae = float(np.abs(sp - ys).mean())
        r = float(np.corrcoef(sp, ys)[0,1])
        return acc, place_mae, a4, smae, r

    for fn, seed in CKPTS:
        path = f'{CKDIR}/{fn}'
        ck = torch.load(path, map_location='cpu')
        ch, bl = ck['channels'], ck['blocks']
        net = ValueMT(ch, bl).to(dev)
        net.load_state_dict(ck['state'])
        nparam = sum(p.numel() for p in net.parameters())
        acc, pmae, a4, smae, r = evaluate(net)
        rec = {'channels': ch, 'blocks': bl, 'params': int(nparam),
               'seed': seed, 'epochs': EPOCHS, 'n_eval': int(nval),
               'fourth_auc': a4, 'place_acc': acc, 'place_mae': pmae,
               'score_r': r, 'score_mae': smae, 'final_epoch': EPOCHS,
               'ckpt': fn}
        jout = f'{JDIR}/{ch}x{bl}_s{seed}.json'
        with open(jout, 'w') as f: json.dump(rec, f, indent=2)
        print(f'{fn}: ch={ch} bl={bl} params={nparam/1e6:.2f}M | '
              f'4th_auc={a4:.4f} place_acc={acc:.4f} score_r={r:+.4f} '
              f'place_mae={pmae:.3f} score_mae={smae:.4f} -> {jout}', flush=True)
        del net; torch.cuda.empty_cache()
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
