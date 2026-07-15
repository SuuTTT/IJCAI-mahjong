"""
train_value.py — multi-task VALUE model on cooked_value.npz.
Shared ResBN trunk (stem+body, GAP) -> 3 heads:
  - place4 : 4-class logits over deal placement (1/2/3/4)  -> CE loss; report acc + expected-place MAE
  - fourth : 1 logit P(this seat gets 4th this deal)        -> BCE; report AUC
  - score  : regress deal_score/48                          -> smooth_l1; report MAE + Pearson r
Trains on GPU, daemon-friendly (flushes log). Held-out 10% split (fixed seed).

  python3 train_value.py --gpu 0 --channels 128 --blocks 20 --epochs 6 --out ckpt/value_mt.pkl
"""
import os, sys, argparse, time
sys.path.insert(0, '/root/IJCAI-mahjong/train/caiest_repro')
import numpy as np, torch, torch.nn as nn

IN_PLANES = 38; GRID = 36

class _BNBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(x + y)

class ValueMT(nn.Module):
    def __init__(self, channels=128, blocks=20):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(IN_PLANES, channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*(_BNBlock(channels) for _ in range(blocks)))
        self.place_head  = nn.Sequential(nn.Linear(channels, 128), nn.ReLU(), nn.Linear(128, 4))
        self.fourth_head = nn.Sequential(nn.Linear(channels, 128), nn.ReLU(), nn.Linear(128, 1))
        self.score_head  = nn.Sequential(nn.Linear(channels, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, obs):                       # obs (B,38,4,9) float
        x = self.body(self.stem(obs))
        x = x.mean(dim=(2, 3))                    # GAP -> (B,ch)
        return self.place_head(x), self.fourth_head(x).squeeze(1), self.score_head(x).squeeze(1)


def auc(y, p):
    # rank-based AUC
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p)+1)
    pos = y == 1; npos = pos.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0: return float('nan')
    return float((ranks[pos].sum() - npos*(npos+1)/2) / (npos*nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/root/IJCAI-mahjong/train/caiest_repro/data/cooked_value.npz')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--channels', type=int, default=128); ap.add_argument('--blocks', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json_out', default='')
    ap.add_argument('--out', default='/root/IJCAI-mahjong/train/caiest_repro/ckpt/value_mt.pkl')
    a = ap.parse_args()
    dev = f'cuda:{a.gpu}'
    torch.cuda.set_device(a.gpu)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    z = np.load(a.data)
    obs = z['obs']                                # int8 (N,38,4,9) keep on CPU, cast per-batch
    place = z['deal_place'].astype(np.int64) - 1  # 0..3
    fourth = (z['deal_place'] == 4).astype(np.float32)
    score = (z['deal_score'].astype(np.float32)) / 48.0
    N = len(place)
    rng = np.random.RandomState(0); idx = rng.permutation(N)   # FIXED split (seed-independent) -> identical held-out across all runs
    nval = N // 10; vi, ti = idx[:nval], idx[nval:]
    assert len(np.intersect1d(vi, ti)) == 0, 'train/val overlap!'
    print(f'N={N} train={len(ti)} val={nval} | place base-rate {[round(float((place==k).mean()),3) for k in range(4)]} | 4th rate {fourth.mean():.3f} | score y mean {score.mean():+.3f} std {score.std():.3f}', flush=True)

    net = ValueMT(a.channels, a.blocks).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    nparam = sum(p.numel() for p in net.parameters())
    print(f'model ch={a.channels} blocks={a.blocks} params={nparam/1e6:.2f}M', flush=True)

    # preload val tensors to GPU in chunks at eval time
    def evaluate():
        net.eval(); pl_logits=[]; f_logits=[]; s_pred=[]
        with torch.no_grad():
            for i in range(0, nval, 4096):
                b = vi[i:i+4096]
                ob = torch.from_numpy(obs[b].astype(np.float32)).to(dev)
                pl, fo, sc = net(ob)
                pl_logits.append(pl.cpu().numpy()); f_logits.append(fo.cpu().numpy()); s_pred.append(sc.cpu().numpy())
        pl = np.concatenate(pl_logits); fo = np.concatenate(f_logits); sp = np.concatenate(s_pred)
        yp = place[vi]; yf = fourth[vi]; ys = score[vi]
        acc = float((pl.argmax(1) == yp).mean())
        # expected placement from softmax -> MAE vs true place(1..4)
        sm = np.exp(pl - pl.max(1, keepdims=True)); sm /= sm.sum(1, keepdims=True)
        exp_place = (sm * np.array([1,2,3,4])).sum(1)
        place_mae = float(np.abs(exp_place - (yp+1)).mean())
        a4 = auc(yf, fo)
        smae = float(np.abs(sp - ys).mean())
        r = float(np.corrcoef(sp, ys)[0,1])
        return acc, place_mae, a4, smae, r

    acc, pmae, a4, smae, r = evaluate()
    print(f'init: place_acc={acc:.4f} place_mae={pmae:.3f} 4th_auc={a4:.4f} score_mae={smae:.4f} score_r={r:+.4f}', flush=True)

    n = len(ti); last = None
    for ep in range(1, a.epochs+1):
        net.train(); perm = np.random.RandomState(1000*a.seed + ep).permutation(n); t0 = time.time(); tot = 0.0; nb = 0
        for i in range(0, n, a.bs):
            b = ti[perm[i:i+a.bs]]
            ob = torch.from_numpy(obs[b].astype(np.float32)).to(dev)
            yp = torch.from_numpy(place[b]).to(dev)
            yf = torch.from_numpy(fourth[b]).to(dev)
            ys = torch.from_numpy(score[b]).to(dev)
            pl, fo, sc = net(ob)
            loss = nn.functional.cross_entropy(pl, yp) \
                 + nn.functional.binary_cross_entropy_with_logits(fo, yf) \
                 + nn.functional.smooth_l1_loss(sc, ys)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        acc, pmae, a4, smae, r = evaluate()
        print(f'ep {ep}/{a.epochs} loss={tot/nb:.4f} {time.time()-t0:.0f}s | place_acc={acc:.4f} place_mae={pmae:.3f} 4th_auc={a4:.4f} score_mae={smae:.4f} score_r={r:+.4f}', flush=True)
        last = {'epoch': ep, 'place_acc': acc, 'place_mae': pmae, 'fourth_auc': a4, 'score_mae': smae, 'score_r': r}
        torch.save({'state': {k: v.cpu() for k,v in net.state_dict().items()},
                    'channels': a.channels, 'blocks': a.blocks},
                   a.out, _use_new_zipfile_serialization=False)
    print(f'DONE -> {a.out}', flush=True)
    if a.json_out:
        import json
        rec = {'channels': a.channels, 'blocks': a.blocks, 'params': int(nparam),
               'seed': a.seed, 'epochs': a.epochs, 'n_eval': int(nval),
               'fourth_auc': last['fourth_auc'], 'place_acc': last['place_acc'],
               'place_mae': last['place_mae'], 'score_r': last['score_r'],
               'score_mae': last['score_mae'], 'final_epoch': last['epoch']}
        with open(a.json_out, 'w') as f: json.dump(rec, f, indent=2)
        print(f'JSON -> {a.json_out}', flush=True)


if __name__ == '__main__':
    main()
