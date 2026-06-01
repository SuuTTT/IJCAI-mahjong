"""
train_bc_v4.py — stronger supervised policy:
  • FAST vectorized suit-permutation augmentation (W/T/B are interchangeable by
    symmetry → 6x effective data, applied on-the-fly via lookup tables). This was
    defined but never wired into earlier training.
  • Bigger network (configurable hidden/blocks).
  • Trains on the 5.1M all-players set (offense + defense), early stopping.
  • Exports fp16 numpy weights ready for the bot.

Usage:
  OPENBLAS_NUM_THREADS=8 python3 train/train_bc_v4.py \
      --data data/processed/official_all.npz \
      --out  train/checkpoints/bc_v4.pt \
      --hidden 1024 --blocks 8 --epochs 60 --batch 4096 --aug
"""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

from train.model import MahjongNet
from train.train_bc_v2 import _suit_perm_act, ALL_PERMS   # reuse action remap
from data.feature_agent import ACT_DIM, OBS_DIM

# ── Build fast lookup tables for suit-permutation augmentation ────────────────
# Tile-index blocks in the observation: W=0..8, T=9..17, B=18..26 (honors 27..33).
SUIT_BLOCKS = [(0, 9), (9, 9), (18, 9)]   # (start, len) for W, T, B

def _tile_lut(perm):
    """256-entry LUT remapping tile-index bytes under a suit permutation.
    Identity for honors (27-33), sentinels (34,35) and empty (255)."""
    lut = np.arange(256, dtype=np.uint8)
    new_start = [SUIT_BLOCKS[p][0] for p in perm]
    for si, (start, ln) in enumerate(SUIT_BLOCKS):
        for k in range(ln):
            lut[start + k] = new_start[si] + k
    return lut

def _act_lut(perm):
    """235-entry action remap LUT under a suit permutation."""
    return np.array([_suit_perm_act(a, perm) for a in range(ACT_DIM)], dtype=np.int64)

# Precompute LUTs for all 6 permutations
_TILE_LUTS = [_tile_lut(p) for p in ALL_PERMS]
_ACT_LUTS  = [_act_lut(p)  for p in ALL_PERMS]
_INV_ACT   = [np.argsort(l) for l in _ACT_LUTS]   # for masks: mask_aug = mask[:, inv]


def augment(obs, mask, act, pi):
    """Apply permutation index pi to a batch (numpy). obs uint8, mask bool, act int64."""
    if pi == 0:   # identity
        return obs, mask, act
    tl, al, inv = _TILE_LUTS[pi], _ACT_LUTS[pi], _INV_ACT[pi]
    o = obs.copy()
    o[:, 2:] = tl[obs[:, 2:]]              # remap tile-index slots (skip winds [0],[1])
    a = al[act]                            # remap chosen action
    m = mask[:, inv]                       # remap legal mask
    return o, m, a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="data/processed/official_all.npz")
    p.add_argument("--out",    default="train/checkpoints/bc_v4.pt")
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch",  type=int, default=4096)
    p.add_argument("--lr",     type=float, default=3e-4)
    p.add_argument("--dropout",type=float, default=0.15)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--aug",    action="store_true", help="enable suit augmentation")
    p.add_argument("--init",   default="", help="warm-start checkpoint (.pt)")
    args = p.parse_args()

    d = np.load(args.data)
    obs, mask, act = d["obs"], d["mask"], d["act"].astype(np.int64)
    N = len(obs)
    rng = np.random.default_rng(42)
    idx = rng.permutation(N); split = int(N*0.9)
    tr, va = idx[:split], idx[split:]
    print(f"data={args.data}  N={N:,}  aug={args.aug}  hidden={args.hidden} blocks={args.blocks}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MahjongNet(hidden=args.hidden, n_blocks=args.blocks, dropout=args.dropout).to(dev)
    if args.init and os.path.exists(args.init):
        ck = torch.load(args.init, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); print(f"warm-start from {args.init}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}  device={dev}")
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=(dev.type=="cuda"))
    ce = nn.CrossEntropyLoss()

    # keep arrays in RAM; index per batch
    def batches(split_idx, shuffle, augment_on):
        order = rng.permutation(split_idx) if shuffle else split_idx
        for i in range(0, len(order), args.batch):
            b = order[i:i+args.batch]
            o, m, a = obs[b], mask[b], act[b]
            if augment_on:
                pi = int(rng.integers(0, len(ALL_PERMS)))
                o, m, a = augment(o, m, a, pi)
            yield (torch.from_numpy(o.astype(np.float32)).to(dev),
                   torch.from_numpy(m).to(dev),
                   torch.from_numpy(a).to(dev))

    best = 1e9; bad = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ep in range(1, args.epochs+1):
        model.train(); tl=0; tc=0; n=0
        for o,m,a in batches(tr, True, args.aug):
            opt.zero_grad()
            with autocast("cuda", enabled=(dev.type=="cuda")):
                logits,_ = model(o, m); loss = ce(logits, a)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tl += loss.item()*len(a); tc += (logits.argmax(1)==a).sum().item(); n += len(a)
        sched.step()
        model.eval(); vl=0; vc=0; vn=0
        with torch.no_grad():
            for o,m,a in batches(va, False, False):
                with autocast("cuda", enabled=(dev.type=="cuda")):
                    logits,_ = model(o, m)
                vl += ce(logits,a).item()*len(a); vc += (logits.argmax(1)==a).sum().item(); vn += len(a)
        vl/=vn; va_acc=vc/vn; ta=tc/n
        mark = "*" if vl<best else " "
        print(f"ep {ep:3d}/{args.epochs} tr_acc={ta:.3f} va_loss={vl:.4f} va_acc={va_acc:.3f} {mark}", flush=True)
        if vl<best:
            best=vl; bad=0
            torch.save({"model":model.state_dict(),"epoch":ep,"val_loss":vl,
                        "val_acc":va_acc,"args":vars(args)}, args.out)
        else:
            bad+=1
            if bad>=args.patience:
                print(f"early stop @ {ep}"); break

    ckpt = torch.load(args.out, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    npz = args.out.replace(".pt","_weights.npz")
    model.export_numpy(npz)
    print(f"best ep={ckpt['epoch']} va_acc={ckpt['val_acc']:.3f}  exported {npz}")


if __name__ == "__main__":
    main()
