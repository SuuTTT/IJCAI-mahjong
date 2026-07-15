"""Obs migration: grow the card one-hot capacity without breaking checkpoints.

The vector observation is [4 scalars][6 tower fracs][8 blocks of N one-hots]
(hand x4, queue x4). Growing N (60 -> 96 in v11) moves the block offsets, so
every kernel row of the vector-branch Dense must move to its new index; rows
for the new (never-hot under old data) slots are zero — making the migrated
network EXACTLY equivalent on all states whose card ids fit the old capacity.

Usage:
    python scripts/migrate_obs.py --old 60 --new 96 --files a.msgpack b.msgpack
    python scripts/migrate_obs.py --old 60 --new 96 --dir /root/ludus_train/league
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from flax.serialization import from_bytes, msgpack_restore, to_bytes

HEAD = 4 + 6          # scalars + tower fractions
BLOCKS = 8            # hand(4) + queue(4)


def index_map(old_n: int, new_n: int) -> np.ndarray:
    """old vector index -> new vector index."""
    idx = list(range(HEAD))
    for b in range(BLOCKS):
        idx.extend(HEAD + b * new_n + c for c in range(old_n))
    return np.asarray(idx)


def migrate_tree(tree, old_n: int, new_n: int) -> tuple[dict, int]:
    old_dim = HEAD + BLOCKS * old_n
    new_dim = HEAD + BLOCKS * new_n
    imap = index_map(old_n, new_n)
    hits = 0

    def walk(node):
        nonlocal hits
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        arr = np.asarray(node)
        if arr.ndim == 2 and arr.shape[0] == old_dim:
            out = np.zeros((new_dim, arr.shape[1]), arr.dtype)
            out[imap] = arr
            hits += 1
            return out
        return node

    return walk(tree), hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=int, required=True)
    ap.add_argument("--new", type=int, required=True)
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()

    files = [Path(f) for f in args.files]
    if args.dir:
        files += sorted(Path(args.dir).glob("*.msgpack"))
    if not files:
        raise SystemExit("nothing to migrate")

    for f in files:
        raw = msgpack_restore(f.read_bytes())
        migrated, hits = migrate_tree(raw, args.old, args.new)
        if hits == 0:
            print(f"SKIP {f} (no ({HEAD + BLOCKS * args.old}, ...) kernel — "
                  f"already migrated or foreign arch)")
            continue
        backup = f.with_suffix(f.suffix + f".pre{args.new}")
        if not backup.exists():
            backup.write_bytes(f.read_bytes())
        f.write_bytes(to_bytes(migrated))
        print(f"OK   {f} ({hits} kernel(s) migrated; backup {backup.name})")


if __name__ == "__main__":
    main()
