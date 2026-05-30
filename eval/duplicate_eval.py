"""
duplicate_eval.py — evaluate bots using the official duplicate (复式) format.

Duplicate rules:
  • 4 fixed tile walls, each played under all 24 seat permutations (4! = 24)
  • Micro-scores for each permutation are summed per seat
  • The 4 micro-score totals are ranked 1–4 → ranking points 4/3/2/1
  • Final standing = sum of ranking points across all walls
  • Tie-break = sum of micro-scores

Usage:
    python3 duplicate_eval.py bot0 bot1 bot2 bot3 \
        [--walls 4] [--seed 42] [--timeout 5] [--jobs 4] [--verbose]

Each botN is a shell command using the one-shot JSON protocol.
"""

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations

# Make run_match importable
sys.path.insert(0, os.path.dirname(__file__))
from run_match import run_match

JUDGE = os.path.realpath(
    os.path.join(os.path.dirname(__file__),
                 "../../workspace/Chinese-Standard-Mahjong/judge/judge")
)


# ── wall generation ────────────────────────────────────────────────────────────

def generate_wall(seed: int) -> str:
    """Generate a shuffled wall string using the same tile set as the judge."""
    tiles = []
    for s in "WBT":
        for n in range(1, 10):
            tiles.extend([f"{s}{n}"] * 4)
    for n in range(1, 5):
        tiles.extend([f"F{n}"] * 4)
    for n in range(1, 4):
        tiles.extend([f"J{n}"] * 4)
    rng = random.Random(seed)
    rng.shuffle(tiles)
    return " ".join(tiles)


# ── scoring helpers ────────────────────────────────────────────────────────────

def micro_to_ranking(micro: list) -> list:
    """Convert 4 micro-scores to 4/3/2/1 ranking points (ties split evenly)."""
    order = sorted(range(4), key=lambda i: micro[i], reverse=True)
    pts = [0.0] * 4
    i = 0
    while i < 4:
        j = i + 1
        while j < 4 and micro[order[j]] == micro[order[i]]:
            j += 1
        pool = sum(4 - k for k in range(i, j))
        share = pool / (j - i)
        for k in range(i, j):
            pts[order[k]] = share
        i = j
    return pts


# ── single wall evaluation ─────────────────────────────────────────────────────

def eval_one_wall(args):
    """Worker: run all 24 permutations of one wall, return per-seat micro totals."""
    bot_cmds, wall_str, quan, timeout, wall_idx = args
    micro = [0.0] * 4   # indexed by logical seat (0=bot0, 1=bot1, …)

    for perm in permutations(range(4)):
        # perm[seat_in_game] = logical bot index
        # We need to map: game seat i → bot_cmds[perm[i]]
        ordered_bots = [bot_cmds[perm[i]] for i in range(4)]
        try:
            result = run_match(ordered_bots, wall_json=wall_str,
                               quan=quan, timeout=timeout)
            game_scores = result["scores"]
            # game_scores[i] is the score for game seat i
            # logical bot perm[i] earned game_scores[i]
            for seat in range(4):
                micro[perm[seat]] += game_scores[seat]
        except Exception as e:
            # A crashed game contributes 0 micro-score for all
            pass

    return wall_idx, micro


# ── main evaluation ────────────────────────────────────────────────────────────

def duplicate_eval(
    bot_cmds: list,
    n_walls: int = 4,
    base_seed: int = 42,
    timeout: float = 5.0,
    quan: int = 0,       # 0=East prevailing wind
    n_jobs: int = 1,
    verbose: bool = False,
) -> dict:

    walls = [(generate_wall(base_seed + i), base_seed + i) for i in range(n_walls)]

    total_ranking = [0.0] * 4
    total_micro   = [0.0] * 4
    per_wall = []

    work_items = [
        (bot_cmds, wall_str, quan, timeout, i)
        for i, (wall_str, _) in enumerate(walls)
    ]

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(eval_one_wall, item): item for item in work_items}
            results = [None] * n_walls
            for f in as_completed(futures):
                wall_idx, micro = f.result()
                results[wall_idx] = micro
    else:
        results = [eval_one_wall(item)[1] for item in work_items]

    for wall_idx, micro in enumerate(results):
        rpts = micro_to_ranking(micro)
        for i in range(4):
            total_ranking[i] += rpts[i]
            total_micro[i]   += micro[i]

        per_wall.append({
            "wall_seed": base_seed + wall_idx,
            "micro": micro,
            "ranking_pts": rpts,
        })

        if verbose:
            print(f"  Wall {wall_idx}  micro={[round(m) for m in micro]}"
                  f"  ranking={[round(r,1) for r in rpts]}", file=sys.stderr)

    # Final standings
    standing = sorted(range(4), key=lambda i: (total_ranking[i], total_micro[i]),
                      reverse=True)

    return {
        "standings": standing,
        "total_ranking_pts": total_ranking,
        "total_micro": total_micro,
        "per_wall": per_wall,
        "n_walls": n_walls,
        "n_games": n_walls * 24,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("bots", nargs=4, metavar="BOT")
    p.add_argument("--walls",   type=int,   default=4)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--quan",    type=int,   default=0)
    p.add_argument("--jobs",    type=int,   default=1)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"Evaluating {args.walls} walls × 24 permutations = {args.walls*24} games …",
          file=sys.stderr)
    t0 = __import__("time").time()

    result = duplicate_eval(
        args.bots,
        n_walls=args.walls,
        base_seed=args.seed,
        timeout=args.timeout,
        quan=args.quan,
        n_jobs=args.jobs,
        verbose=args.verbose,
    )

    elapsed = __import__("time").time() - t0
    print(f"Done in {elapsed:.1f}s", file=sys.stderr)

    # Pretty summary
    print("\n=== Duplicate Evaluation Results ===")
    print(f"{'Bot':<6} {'Ranking pts':>12} {'Micro score':>12} {'Rank':>6}")
    for i, sid in enumerate(result["standings"]):
        print(f"  bot{sid}  "
              f"{result['total_ranking_pts'][sid]:>10.1f}  "
              f"{result['total_micro'][sid]:>10.0f}  "
              f"  #{i+1}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
