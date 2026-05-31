"""
train_bc_v2.py — improved BC training with:
  - Dropout regularisation (avoids overfitting on 192k samples)
  - Suit-permutation augmentation (6× free data from W/B/T symmetry)
  - Early stopping (patience=10)
  - Mixed precision (fp16 on GPU)

Usage:
    python3 train/train_bc_v2.py \
        --data data/processed/official_winner.npz \
        --out  train/checkpoints/bc_v2.pt \
        [--data2 data/processed/official_all.npz]   # optionally stack datasets
"""

import argparse, os, sys, numpy as np, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.cuda.amp import autocast, GradScaler

from train.model import MahjongNet
from data.feature_agent import OBS_DIM, ACT_DIM, ACT, TILE_LIST

# ── Suit permutation augmentation ─────────────────────────────────────────────
# Tiles W1-W9=idx 0-8, T1-T9=9-17, B1-B9=18-26 (TILE_LIST order)
# Permuting suit labels (W↔T↔B) is a valid symmetry for MCR.

SUIT_BLOCKS = [(0, 9), (9, 9), (18, 9)]   # (start_idx, length) for W, T, B

def _suit_perm_obs(obs: np.ndarray, perm: tuple) -> np.ndarray:
    """
    Permute suits W/T/B in the obs vector.
    perm is a permutation of (0,1,2) mapping original to new suit position.
    E.g. perm=(1,0,2) means swap W and T.
    """
    o = obs.copy()
    # The obs uses tile INDEX values in hand, wall, history, meld slots.
    # We need to remap any byte that falls in W/T/B tile index ranges.
    new_start = [SUIT_BLOCKS[p][0] for p in perm]

    def remap(byte_val):
        if byte_val == 255:   # empty slot
            return byte_val
        for si, (start, length) in enumerate(SUIT_BLOCKS):
            if start <= byte_val < start + length:
                return new_start[si] + (byte_val - start)
        return byte_val

    remap_vec = np.arange(256, dtype=np.uint8)
    for i in range(256):
        remap_vec[i] = remap(i)

    # Apply to all relevant byte positions in obs
    # (everything except PREVALENT_WIND [0] and SEAT_WIND [1])
    o[2:] = remap_vec[o[2:]]
    return o


def _suit_perm_act(act_idx: int, perm: tuple) -> int:
    """Remap an action index under a suit permutation."""
    new_start = [SUIT_BLOCKS[p][0] for p in perm]

    def remap_tile(ti):
        for si, (start, length) in enumerate(SUIT_BLOCKS):
            if start <= ti < start + length:
                return new_start[si] + (ti - start)
        return ti

    # Play: act_idx in [ACT["Play"], ACT["Play"]+34)
    if ACT["Play"] <= act_idx < ACT["Chi"]:
        return ACT["Play"] + remap_tile(act_idx - ACT["Play"])
    # Chi: more complex, remap tile within suit
    if ACT["Chi"] <= act_idx < ACT["Peng"]:
        chi_off = act_idx - ACT["Chi"]
        suit_idx = chi_off // 21
        rem      = chi_off % 21
        new_suit = perm[suit_idx] if suit_idx < 3 else suit_idx
        return ACT["Chi"] + new_suit * 21 + rem
    # Peng, Gang, AnGang, BuGang: remap tile
    for act_name, act_off in [("Peng",ACT["Peng"]),("Gang",ACT["Gang"]),
                               ("AnGang",ACT["AnGang"]),("BuGang",ACT["BuGang"])]:
        next_off = ACT_DIM if act_off == ACT["BuGang"] else {
            ACT["Peng"]:ACT["Gang"], ACT["Gang"]:ACT["AnGang"],
            ACT["AnGang"]:ACT["BuGang"]
        }[act_off]
        if act_off <= act_idx < next_off:
            return act_off + remap_tile(act_idx - act_off)
    return act_idx  # Pass, Hu: unchanged


ALL_PERMS = [
    (0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)
]

def augment_batch(obs_np, mask_np, act_np, n_perms=6):
    """Return augmented (obs, mask, act) with up to n_perms suit permutations."""
    perms = random.sample(ALL_PERMS, min(n_perms, len(ALL_PERMS)))
    aug_obs, aug_mask, aug_act = [obs_np], [mask_np], [act_np]
    for perm in perms[1:]:   # skip identity (0,1,2)
        new_obs  = np.array([_suit_perm_obs(o, perm) for o in obs_np])
        new_act  = np.array([_suit_perm_act(int(a), perm) for a in act_np], dtype=np.int16)
        # mask: remap same as act but for all 235 positions
        new_mask = np.zeros_like(mask_np)
        for i, m in enumerate(mask_np):
            for j in range(ACT_DIM):
                if m[j]:
                    nj = _suit_perm_act(j, perm)
                    if 0 <= nj < ACT_DIM:
                        new_mask[i, nj] = True
        aug_obs.append(new_obs)
        aug_mask.append(new_mask)
        aug_act.append(new_act)
    return (np.concatenate(aug_obs),
            np.concatenate(aug_mask),
            np.concatenate(aug_act))


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_npz(path):
    d = np.load(path)
    return d["obs"], d["mask"], d["act"].astype(np.int64)


# ── Training ───────────────────────────────────────────────────────────────────

def train(args):
    obs, mask, act = load_npz(args.data)

    if args.data2 and os.path.exists(args.data2):
        o2, m2, a2 = load_npz(args.data2)
        obs  = np.concatenate([obs, o2])
        mask = np.concatenate([mask, m2])
        act  = np.concatenate([act, a2.astype(np.int64)])
        print(f"Stacked datasets: {len(obs):,} total samples")

    N = len(obs)
    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    split = int(N * 0.9)
    tr_i, va_i = idx[:split], idx[split:]

    def make_loader(i, shuffle):
        ds = TensorDataset(
            torch.from_numpy(obs[i].astype(np.float32)),
            torch.from_numpy(mask[i]),
            torch.from_numpy(act[i]),
        )
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle,
                          num_workers=4, pin_memory=True)

    tr_loader = make_loader(tr_i, True)
    va_loader = make_loader(va_i, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  {len(tr_i):,} train / {len(va_i):,} val")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model  = MahjongNet(hidden=args.hidden, n_blocks=args.blocks, dropout=args.dropout).to(device)
    opt    = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler(enabled=(device.type=="cuda"))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}  (hidden={args.hidden} blocks={args.blocks})")

    best_val  = float("inf")
    patience  = args.patience
    no_improv = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_loss = tr_correct = 0
        for o_b, m_b, a_b in tr_loader:
            o_b = o_b.to(device); m_b = m_b.to(device); a_b = a_b.to(device)
            opt.zero_grad()
            with autocast(enabled=(device.type=="cuda")):
                logits, _ = model(o_b, m_b)
                loss = nn.CrossEntropyLoss()(logits, a_b)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tr_loss    += loss.item() * len(a_b)
            tr_correct += (logits.argmax(1) == a_b).sum().item()

        sched.step()
        tr_loss /= len(tr_i); tr_acc = tr_correct / len(tr_i)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        va_loss = va_correct = 0
        with torch.no_grad():
            for o_b, m_b, a_b in va_loader:
                o_b = o_b.to(device); m_b = m_b.to(device); a_b = a_b.to(device)
                with autocast(enabled=(device.type=="cuda")):
                    logits, _ = model(o_b, m_b)
                va_loss    += nn.CrossEntropyLoss()(logits, a_b).item() * len(a_b)
                va_correct += (logits.argmax(1) == a_b).sum().item()
        va_loss /= len(va_i); va_acc = va_correct / len(va_i)

        mark = "*" if va_loss < best_val else " "
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.3f} {mark}")

        if va_loss < best_val:
            best_val = va_loss; no_improv = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": va_loss, "args": vars(args)}, args.out)
        else:
            no_improv += 1
            if no_improv >= patience:
                print(f"Early stop at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nBest: epoch {torch.load(args.out,map_location='cpu',weights_only=False)['epoch']}  val_loss={best_val:.4f}")

    # Export numpy weights
    ckpt  = torch.load(args.out, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    npz   = args.out.replace(".pt", "_weights.npz")
    model.export_numpy(npz)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/processed/official_winner.npz")
    p.add_argument("--data2",   default="",      help="optional second dataset to stack")
    p.add_argument("--out",     default="train/checkpoints/bc_v2.pt")
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--batch",   type=int,   default=4096)
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--hidden",  type=int,   default=512)
    p.add_argument("--blocks",  type=int,   default=6)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--patience",type=int,   default=15)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
