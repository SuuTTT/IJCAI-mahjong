"""
train_bc.py — Behavior Cloning (supervised learning) from game logs.

Usage:
    python3 train/train_bc.py \
        --data data/processed/selfplay.npz \
        --out  train/checkpoints/bc_v1.pt \
        [--epochs 50] [--batch 2048] [--lr 3e-4] [--hidden 512] [--blocks 6]

After training, export numpy weights for serve-time inference:
    python3 train/train_bc.py --export train/checkpoints/bc_v1.pt \
        --npz train/checkpoints/bc_v1_weights.npz
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed. Install with: pip3 install torch --index-url https://download.pytorch.org/whl/cpu")


def load_dataset(path):
    data = np.load(path)
    obs  = data["obs"].astype(np.float32)   # (N, 240)
    mask = data["mask"].astype(bool)         # (N, 235)
    act  = data["act"].astype(np.int64)      # (N,)
    print(f"Loaded {len(obs)} samples from {path}")
    return obs, mask, act


def train(args):
    if not HAS_TORCH:
        print("Cannot train without PyTorch.")
        return

    from train.model import MahjongNet

    obs, mask, act = load_dataset(args.data)
    N = len(obs)
    split = int(N * 0.9)
    idx = np.random.permutation(N)
    tr_idx, va_idx = idx[:split], idx[split:]

    def make_loader(i, shuffle):
        ds = TensorDataset(
            torch.tensor(obs[i]),
            torch.tensor(mask[i]),
            torch.tensor(act[i]),
        )
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle,
                          num_workers=2, pin_memory=torch.cuda.is_available())

    tr_loader = make_loader(tr_idx, True)
    va_loader = make_loader(va_idx, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}, {len(tr_idx)} train / {len(va_idx)} val samples")

    model = MahjongNet(hidden=args.hidden, n_blocks=args.blocks).to(device)
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        for o, m, a in tr_loader:
            o, m, a = o.to(device), m.to(device), a.to(device)
            logits, _ = model(o, m)
            loss = nn.CrossEntropyLoss()(logits, a)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss    += loss.item() * len(a)
            tr_correct += (logits.argmax(1) == a).sum().item()

        sched.step()
        tr_loss    /= len(tr_idx)
        tr_acc      = tr_correct / len(tr_idx)

        model.eval()
        va_loss = 0.0
        va_correct = 0
        with torch.no_grad():
            for o, m, a in va_loader:
                o, m, a = o.to(device), m.to(device), a.to(device)
                logits, _ = model(o, m)
                va_loss    += nn.CrossEntropyLoss()(logits, a).item() * len(a)
                va_correct += (logits.argmax(1) == a).sum().item()
        va_loss /= len(va_idx)
        va_acc   = va_correct / len(va_idx)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.3f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": va_loss,
                        "args": vars(args)},
                       args.out)

    print(f"Best checkpoint at {args.out}  (val_loss={best_val:.4f})")

    # Auto-export numpy weights
    npz = args.out.replace(".pt", "_weights.npz")
    ckpt = torch.load(args.out, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.export_numpy(npz)
    print(f"Numpy weights saved to {npz}")


def export_only(args):
    if not HAS_TORCH:
        return
    from train.model import MahjongNet
    ckpt  = torch.load(args.export, map_location="cpu")
    model = MahjongNet(hidden=ckpt["args"].get("hidden",512),
                       n_blocks=ckpt["args"].get("blocks",6))
    model.load_state_dict(ckpt["model"])
    model.export_numpy(args.npz)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/processed/selfplay.npz")
    p.add_argument("--out",     default="train/checkpoints/bc_v1.pt")
    p.add_argument("--epochs",  type=int,   default=50)
    p.add_argument("--batch",   type=int,   default=2048)
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--hidden",  type=int,   default=512)
    p.add_argument("--blocks",  type=int,   default=6)
    p.add_argument("--export",  help="PT checkpoint to export numpy weights from")
    p.add_argument("--npz",     help="Output .npz path for --export")
    args = p.parse_args()

    if args.export:
        export_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
