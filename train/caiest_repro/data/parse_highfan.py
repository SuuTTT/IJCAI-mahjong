"""
parse_highfan.py — extract WINNER trajectories only from HIGH-FAN games, to teach
8-fan+ hand construction (conversion). Reuses parse_official's per-game parser; adds a
winner-fan filter read from the explicit `Fan <N> ...` line that precedes each win's
`Score` line in data.txt.

  python3 data/parse_highfan.py --min-fan 12 --out data/processed/official_hf12.npz
  python3 data/parse_highfan.py --hist     # just print the fan distribution
"""
import os, sys, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import Counter
from data.parse_official import iter_games, parse_one_game

FAN_RE = re.compile(r"^Fan (\d+)\b")

def game_fan(game_lines):
    """Winner fan = the Fan N line preceding the final non-zero Score. -1 if Huang."""
    for i in range(len(game_lines) - 1, -1, -1):
        m = FAN_RE.match(game_lines[i].strip())
        if m:
            return int(m.group(1))
    return -1

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="data/raw/data.zip")
    p.add_argument("--out", default="data/processed/official_highfan.npz")
    p.add_argument("--min-fan", type=int, default=12)
    p.add_argument("--max-games", type=int, default=0)
    p.add_argument("--hist", action="store_true")
    args = p.parse_args()

    if args.hist:
        h = Counter(); tot = 0
        for gl in iter_games(args.inp):
            f = game_fan(gl); tot += 1
            h[f if f < 0 else min(f, 40)] += 1
        print(f"total games: {tot}")
        print(f"  Huang (draw): {h[-1]} ({100*h[-1]/tot:.0f}%)")
        wins = tot - h[-1]
        for thr in (8, 10, 12, 14, 16, 20, 24):
            n = sum(c for f, c in h.items() if f >= thr)
            print(f"  fan>={thr:2d}: {n:7d} games ({100*n/max(1,wins):.0f}% of wins, {100*n/tot:.0f}% of all)")
        return

    obs_l, mask_l, act_l = [], [], []
    kept = skipped = errs = 0
    for gl in iter_games(args.inp):
        if args.max_games and kept >= args.max_games: break
        try:
            if game_fan(gl) < args.min_fan:
                skipped += 1; continue
            s = parse_one_game(gl, winner_only=True, all_players=False)
            if not s:
                skipped += 1; continue
            for o, m, a in s:
                obs_l.append(o); mask_l.append(m); act_l.append(a)
            kept += 1
        except Exception:
            errs += 1
        if (kept + skipped) % 10000 == 0:
            print(f"  kept {kept} games / {len(obs_l)} samples / skipped {skipped} / errs {errs}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out,
                        obs=np.asarray(obs_l, dtype=np.uint8),
                        mask=np.asarray(mask_l, dtype=np.bool_),
                        act=np.asarray(act_l, dtype=np.int16))
    print(f"DONE: {kept} games, {len(obs_l)} samples (min-fan={args.min_fan}) -> {args.out}")

if __name__ == "__main__":
    main()
