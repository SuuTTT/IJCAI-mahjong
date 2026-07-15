"""
awr_critic.py — Advantage-Weighted Regression toward SPECIAL POINTS, using the VALUE model
as critic. Finetune the moyu actor (ResBNCNN 128x40) on cooked_value decisions, weighting each
sample by w = clip(exp(beta * A), wlo, whi), where A is the cached, baseline-subtracted advantage
A = special_points(realized place) - V_place(s)  (data/adv_cache.npz, aligned to cooked_value).

KL leash to moyu: low LR + a fraction `mix` of batches that are PLAIN BC toward moyu's own argmax
(weight 1), keeping the updated policy near moyu rather than drifting to the dataset behaviour.

Saves a fused (torch-1.4-safe) state_dict loadable by frontier_gate with
  --cand-kind resbn_fused --cand-cfg channels=128,blocks=40

  python3 awr_critic.py --beta 1.0 --steps 8000 --mix 0.3 --out ckpt/rl/moyu_critic_b1.pkl
"""
import os, sys, time, argparse
sys.path.insert(0, '/root/IJCAI-mahjong/train/caiest_repro')
import numpy as np, torch, torch.nn.functional as F
from models_explore import ResBNCNN, fuse_resbn
from suit_aug import PERMS, action_perm, fwd_action_perm

DATA = '/root/IJCAI-mahjong/train/caiest_repro/data/cooked_value.npz'
ADV = '/root/IJCAI-mahjong/train/caiest_repro/data/adv_cache.npz'
MOYU = '/root/assets/moyu_bn_128x40.pkl'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', default=MOYU)
    ap.add_argument('--data', default=DATA); ap.add_argument('--adv', default=ADV)
    ap.add_argument('--channels', type=int, default=128); ap.add_argument('--blocks', type=int, default=40)
    ap.add_argument('--steps', type=int, default=8000); ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--lr', type=float, default=5e-5)        # low LR = part of the KL leash
    ap.add_argument('--aug', type=float, default=0.8)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--wlo', type=float, default=0.0); ap.add_argument('--whi', type=float, default=10.0)
    ap.add_argument('--mix', type=float, default=0.3)        # frac of batches = plain BC->moyu (leash)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=1)   # multi-seed E4: seeds data RNG + torch
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = f'cuda:{a.gpu}'; torch.cuda.set_device(a.gpu)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    z = np.load(a.data)
    O, M, A = z['obs'], z['mask'], z['act'].astype(np.int64)
    adv = np.load(a.adv)['adv'].astype(np.float64)
    assert len(adv) == len(A), f'adv {len(adv)} != data {len(A)}'
    W = np.clip(np.exp(a.beta * adv), a.wlo, a.whi).astype(np.float32)
    print(f'N={len(A)} beta={a.beta} | adv mean={adv.mean():+.4f} std={adv.std():.4f}', flush=True)
    print(f'weight: min={W.min():.3f} max={W.max():.3f} mean={W.mean():.3f} '
          f'std={W.std():.3f} corr(w,adv)={np.corrcoef(W,adv)[0,1]:.3f} '
          f'frac_at_whi={(W>=a.whi-1e-6).mean():.3f}', flush=True)
    Ot, Mt, At, Wt = (torch.from_numpy(O), torch.from_numpy(M), torch.from_numpy(A), torch.from_numpy(W))

    # actor + frozen moyu reference (for the plain-BC leash target)
    m = ResBNCNN(channels=a.channels, blocks=a.blocks).to(dev)
    sd = torch.load(a.init, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
    m.load_state_dict(sd)
    ref = ResBNCNN(channels=a.channels, blocks=a.blocks).to(dev)
    ref.load_state_dict(sd); ref.eval()
    for p in ref.parameters(): p.requires_grad_(False)
    print(f'loaded init {a.init} (+ frozen moyu reference for leash)', flush=True)

    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    rows = [torch.tensor([p[0],p[1],p[2],3], device=dev) for p in PERMS]
    Amaps = [torch.tensor(action_perm(p), device=dev, dtype=torch.long) for p in PERMS]
    Fmaps = [torch.tensor(fwd_action_perm(p), device=dev, dtype=torch.long) for p in PERMS]
    rng = np.random.RandomState(a.seed); n = len(At)
    m.train(); t0 = time.time()
    for s in range(a.steps):
        b = rng.randint(0, n, a.bs)
        o = Ot[b].to(dev); mk = Mt[b].float().to(dev)
        leash = rng.random() < a.mix
        if a.aug > 0 and rng.random() < a.aug:
            pi = rng.randint(1, 6); o = o[:, :, rows[pi], :]; mk = mk[:, Amaps[pi]]
            y_data = Fmaps[pi][At[b].to(dev)]
        else:
            y_data = At[b].to(dev)
        if leash:
            with torch.no_grad():
                rl = ref({'is_training': False, 'obs': {'observation': o, 'action_mask': mk}})
                y = rl.argmax(1)
            w = torch.ones(a.bs, device=dev)
        else:
            y = y_data
            w = Wt[b].to(dev)
        with torch.cuda.amp.autocast(enabled=True):
            logits = m({'is_training': True, 'obs': {'observation': o, 'action_mask': mk}})
            ce = F.cross_entropy(logits, y, reduction='none')
            loss = (w * ce).sum() / (w.sum() + 1e-6)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if (s+1) % 1000 == 0:
            print(f'step{s+1}/{a.steps} loss={loss.item():.3f} leash={leash} ({time.time()-t0:.0f}s)', flush=True)
    fused = fuse_resbn(m.cpu().eval())
    torch.save(fused.state_dict(), a.out, _use_new_zipfile_serialization=False)
    print(f'DONE -> {a.out} (fused)', flush=True)

if __name__ == '__main__':
    main()
