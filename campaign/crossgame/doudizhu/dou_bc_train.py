"""Behavior-cloning trainer for Doudizhu rule agent. MLP over obs(790)->27472 logits,
masked to legal, label-smoothed CE toward chosen action id. Saves fused state_dict .pkl."""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from dou_model import DouMLP, make_mask, masked_logits, N_ACT

def smoothed_ce(logits, mask, y, eps):
    """CE with label smoothing spread over the LEGAL set of each sample."""
    ml = masked_logits(logits, mask)
    logp = F.log_softmax(ml, dim=1)
    nll = -logp.gather(1, y.view(-1, 1)).squeeze(1)
    legal = mask.float()
    logp_safe = torch.where(mask, logp, torch.zeros_like(logp))
    mean_legal = logp_safe.sum(1) / legal.sum(1).clamp(min=1.0)
    return ((1 - eps) * nll + eps * (-mean_legal)).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dou_data.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lsm", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--valevery", type=int, default=5000)
    ap.add_argument("--nval", type=int, default=20000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    d = np.load(a.data)
    obs, act, legal = d["obs"], d["act"].astype(np.int64), d["legal"]
    N = len(act)
    rng = np.random.RandomState(12345); perm = rng.permutation(N)
    nval = min(a.nval, N // 5); vidx = np.sort(perm[:nval]); tidx = perm[nval:]
    print(f"N={N:,} train={len(tidx):,} val={len(vidx):,} hidden={a.hidden} layers={a.layers} "
          f"seed={a.seed} steps={a.steps} lr={a.lr} bs={a.bs}", flush=True)

    obs_g = torch.from_numpy(obs).to(dev)                # int8 (N,790)
    legal_g = torch.from_numpy(legal).to(dev)            # int32 (N,MAXLEGAL)
    act_g = torch.from_numpy(act).to(dev)                # int64 (N,)

    net = DouMLP(hidden=a.hidden, layers=a.layers).to(dev)
    print(f"params {sum(p.numel() for p in net.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = torch.cuda.amp.GradScaler()

    def fetch(idx):
        idx_t = torch.as_tensor(idx, device=dev)
        ob = obs_g[idx_t].float()
        mk = make_mask(legal_g[idx_t])
        y = act_g[idx_t]
        return ob, mk, y

    @torch.no_grad()
    def val_acc():
        net.eval(); c = 0
        for i in range(0, len(vidx), 8192):
            b = vidx[i:i + 8192]
            ob, mk, y = fetch(b)
            pr = masked_logits(net(ob), mk).argmax(1)
            c += (pr == y).sum().item()
        net.train(); return c / len(vidx)

    r2 = np.random.RandomState(1 + a.seed); nt = len(tidx); best = 0.0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time(); net.train()
    for s in range(a.steps):
        b = tidx[r2.randint(0, nt, a.bs)]
        ob, mk, y = fetch(b)
        with torch.cuda.amp.autocast():
            logits = net(ob)
            loss = smoothed_ce(logits, mk, y, a.lsm)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if (s % a.valevery == 0 and s > 0) or s == 200:
            v = val_acc()
            if v > best:
                best = v
                torch.save(net.state_dict(), a.out, _use_new_zipfile_serialization=False)
            print(f"  step {s}/{a.steps} loss {loss.item():.4f} val {v:.4f} best {best:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    v = val_acc()
    if v > best:
        best = v; torch.save(net.state_dict(), a.out, _use_new_zipfile_serialization=False)
    if not os.path.exists(a.out):
        torch.save(net.state_dict(), a.out, _use_new_zipfile_serialization=False)
    print(f"DONE best_val={best:.4f} -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
