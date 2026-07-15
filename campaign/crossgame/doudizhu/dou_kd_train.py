"""KD-from-ensemble trainer. loss = alpha*CE(student, mean-softmax-over-legal of N teacher pkls)
                                 + (1-alpha)*label-smoothed CE(action). alpha=0.7."""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from dou_model import DouMLP, make_mask, masked_logits, N_ACT

def smoothed_ce(logits, mask, y, eps):
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
    ap.add_argument("--teachers", required=True, help="comma-separated .pkl paths")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.7)
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
    obs_g = torch.from_numpy(obs).to(dev)
    legal_g = torch.from_numpy(legal).to(dev)
    act_g = torch.from_numpy(act).to(dev)

    net = DouMLP(hidden=a.hidden, layers=a.layers).to(dev)
    teachers = []
    for tp in a.teachers.split(","):
        t = DouMLP(hidden=a.hidden, layers=a.layers)
        t.load_state_dict(torch.load(tp, map_location="cpu")); t.eval().to(dev)
        for p_ in t.parameters(): p_.requires_grad_(False)
        teachers.append(t)
    print(f"N={N:,} train={len(tidx):,} val={len(vidx):,} KD teachers={len(teachers)} "
          f"alpha={a.alpha} seed={a.seed} steps={a.steps}", flush=True)
    print(f"params {sum(p.numel() for p in net.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = torch.cuda.amp.GradScaler()

    def fetch(idx):
        idx_t = torch.as_tensor(idx, device=dev)
        return obs_g[idx_t].float(), make_mask(legal_g[idx_t]), act_g[idx_t]

    @torch.no_grad()
    def val_acc():
        net.eval(); c = 0
        for i in range(0, len(vidx), 8192):
            b = vidx[i:i + 8192]; ob, mk, y = fetch(b)
            c += (masked_logits(net(ob), mk).argmax(1) == y).sum().item()
        net.train(); return c / len(vidx)

    r2 = np.random.RandomState(1 + a.seed); nt = len(tidx); best = 0.0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time(); net.train()
    for s in range(a.steps):
        b = tidx[r2.randint(0, nt, a.bs)]; ob, mk, y = fetch(b)
        with torch.no_grad():
            tacc = None
            for t in teachers:
                p = torch.softmax(masked_logits(t(ob), mk), 1)
                tacc = p if tacc is None else tacc + p
            tsoft = tacc / len(teachers)
        with torch.cuda.amp.autocast():
            logits = net(ob)
        logp = F.log_softmax(masked_logits(logits, mk), 1)
        kd = -(tsoft * logp).sum(1).mean()
        loss = a.alpha * kd + (1 - a.alpha) * smoothed_ce(logits, mk, y, a.lsm)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if (s % a.valevery == 0 and s > 0) or s == 200:
            v = val_acc()
            if v > best:
                best = v; torch.save(net.state_dict(), a.out, _use_new_zipfile_serialization=False)
            print(f"  step {s}/{a.steps} loss {loss.item():.4f} kd {kd.item():.4f} val {v:.4f} "
                  f"best {best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    v = val_acc()
    if v > best:
        best = v; torch.save(net.state_dict(), a.out, _use_new_zipfile_serialization=False)
    if not os.path.exists(a.out):
        torch.save(net.state_dict(), a.out, _use_new_zipfile_serialization=False)
    print(f"DONE best_val={best:.4f} -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
