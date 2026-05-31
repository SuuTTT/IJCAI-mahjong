"""
parse_official.py — parse the official strong AI dataset (data.txt) into
(obs[240], valid_mask[235], action_idx) .npz training data.

The data format (from README-en.txt):
  Match <ID>
  Wind <W>
  Player 0 Deal t1 t2 ... t13
  Player 1 Deal ...
  Player 2 Deal ...
  Player 3 Deal ...
  Player <N> Draw <tile>
  Player <N> Play <tile>      [may have "Ignore Player X CHI/PENG/GANG/HU tile"]
  Player <N> Chi <mid>
  Player <N> Peng <tile>
  Player <N> Gang <tile>
  Player <N> AnGang <tile>
  Player <N> BuGang <tile>
  Player <N> Hu <tile>
  Fan <F> <desc>              (if someone wins)
  Score <s0> <s1> <s2> <s3>
  OR
  Huang
  Score 0 0 0 0

Usage:
    python3 data/parse_official.py \
        --inp data/raw/data.zip \
        --out data/processed/official.npz \
        [--max-games 50000] [--winner-only] [--all-players]

The strongest signal comes from winner's decisions (--winner-only).
Using --all-players gives 4x data but includes losing strategies.
"""

import argparse
import os
import sys
import zipfile
import re
import numpy as np
from typing import List, Optional, Tuple, Iterator

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.feature_agent import (
    FeatureAgent, ACT, ACT_DIM, OBS_DIM, TILE_INDEX, TILE_LIST, chi_action_idx
)

# ── Game parser ────────────────────────────────────────────────────────────────

def iter_games(path: str) -> Iterator[List[str]]:
    """Yield one game as a list of lines."""
    if path.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        fname = next(n for n in zf.namelist() if n == "data.txt" or n.endswith("data.txt"))
        fh = zf.open(fname)
        lines_iter = (l.decode("utf-8", "replace").rstrip() for l in fh)
    else:
        lines_iter = (l.rstrip() for l in open(path, encoding="utf-8", errors="replace"))

    game = []
    for line in lines_iter:
        if line.startswith("Match ") and game:
            yield game
            game = [line]
        else:
            game.append(line)
    if game:
        yield game


def parse_score_line(line: str) -> Optional[List[int]]:
    m = re.match(r"Score\s+([-\d]+)\s+([-\d]+)\s+([-\d]+)\s+([-\d]+)", line)
    if m:
        return [int(m.group(i)) for i in range(1, 5)]
    return None


def game_winner(game_lines: List[str]) -> int:
    """Return winning player index (-1 for 荒牌)."""
    for i in range(len(game_lines) - 1, -1, -1):
        scores = parse_score_line(game_lines[i])
        if scores:
            return next((j for j, s in enumerate(scores) if s > 0), -1)
    return -1


# ── Action encoder ────────────────────────────────────────────────────────────

def encode_action(action: str, tile: Optional[str],
                  cur_discard_tile: Optional[str],
                  player: int, agent: FeatureAgent) -> int:
    """Encode a player's action as an action index."""
    if action == "Hu":                          return ACT["Hu"]
    if action == "Play" and tile:
        return ACT["Play"] + TILE_INDEX.get(tile, 0)
    if action == "Peng":                        return ACT["Peng"] + TILE_INDEX.get(tile or cur_discard_tile or "W1", 0)
    if action == "Gang":                        return ACT["Gang"] + TILE_INDEX.get(tile or cur_discard_tile or "W1", 0)
    if action == "AnGang" and tile:             return ACT["AnGang"] + TILE_INDEX.get(tile, 0)
    if action == "BuGang" and tile:             return ACT["BuGang"] + TILE_INDEX.get(tile, 0)
    if action == "Chi" and tile and cur_discard_tile:
        suit, mid_n = tile[0], int(tile[1])
        dis_n = int(cur_discard_tile[1]) if cur_discard_tile[0] == suit else mid_n
        return chi_action_idx(suit, mid_n, dis_n)
    return ACT["Pass"]


# ── Main conversion ────────────────────────────────────────────────────────────

def parse_one_game(game_lines: List[str],
                   winner_only: bool = True,
                   all_players: bool = False
                   ) -> List[Tuple[np.ndarray, np.ndarray, int]]:

    winner = game_winner(game_lines)
    if winner_only and winner == -1:
        return []

    # Parse header
    wind = 0
    deals = {}
    start = 0
    for i, line in enumerate(game_lines):
        if line.startswith("Wind "):
            wind = int(line.split()[1])
        elif line.startswith("Player ") and "Deal " in line:
            parts = line.split()
            pid   = int(parts[1])
            tiles = parts[3:]
            deals[pid] = tiles
        if len(deals) == 4:
            start = i + 1
            break

    if len(deals) < 4:
        return []

    agents = [FeatureAgent(p) for p in range(4)]
    for p in range(4):
        agents[p].update(f"Wind {wind}")
        agents[p].update("Deal " + " ".join(deals[p]))

    samples = []
    last_discard: Optional[str] = None

    target_pids = set(range(4)) if all_players else ({winner} if winner >= 0 else set())

    for line in game_lines[start:]:
        line = line.strip()
        if not line or line.startswith("Match ") or line.startswith("Wind "):
            break
        if line.startswith("Fan ") or line.startswith("Score ") or line == "Huang":
            break

        # Strip "Ignore ..." suffixes
        line_main = re.split(r"\s+Ignore\s+", line)[0].strip()
        ignored   = re.findall(r"Ignore\s+(.*?)(?=\s+Ignore|$)", line)

        # Parse main action
        m = re.match(r"Player (\d) (\w+)(?: (.+))?$", line_main)
        if not m:
            continue
        pid    = int(m.group(1))
        action = m.group(2)
        tile   = m.group(3).strip() if m.group(3) else None

        # Get obs + valid BEFORE action
        obs_copy   = agents[pid].obs.copy()
        valid_copy = list(agents[pid].valid)

        # Is this a decision turn for our target player?
        is_decision = (
            pid in target_pids and
            action in ("Play", "Hu", "Chi", "Peng", "Gang", "AnGang", "BuGang") and
            len(valid_copy) > 1
        )

        if is_decision:
            act_idx   = encode_action(action, tile, last_discard, pid, agents[pid])
            valid_mask = np.zeros(ACT_DIM, dtype=np.bool_)
            for v in valid_copy:
                if 0 <= v < ACT_DIM:
                    valid_mask[v] = True
            if act_idx in valid_copy:
                samples.append((obs_copy, valid_mask, act_idx))

        # Update all agents
        if action == "Draw":
            # Drawing player gets "Draw tile" (adds tile to hand, computes valid)
            # Other players get "Player N Draw" (just tracks wall count)
            agents[pid].update(f"Draw {tile}" if tile else f"Player {pid} Draw")
            for p in range(4):
                if p != pid:
                    agents[p].update(f"Player {pid} Draw")
        elif action == "Play":
            last_discard = tile
            for p in range(4):
                agents[p].update(f"Player {pid} Play {tile}")
        elif action == "Peng":
            for p in range(4):
                agents[p].update(f"Player {pid} Peng")
        elif action == "Chi":
            for p in range(4):
                agents[p].update(f"Player {pid} Chi {tile}")
        elif action == "Gang":
            for p in range(4):
                agents[p].update(f"Player {pid} Gang")
        elif action == "AnGang":
            for p in range(4):
                agents[p].update(f"Player {pid} AnGang {tile}")
        elif action == "BuGang":
            for p in range(4):
                agents[p].update(f"Player {pid} BuGang {tile}")
        elif action == "Hu":
            break

    return samples


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inp",         default="data/raw/data.zip")
    p.add_argument("--out",         default="data/processed/official.npz")
    p.add_argument("--max-games",   type=int, default=0,  help="0=all")
    p.add_argument("--winner-only", action="store_true", default=True)
    p.add_argument("--all-players", action="store_true")
    args = p.parse_args()

    winner_only = not args.all_players
    obs_list, mask_list, act_list = [], [], []
    done = errs = skipped = 0

    for game_lines in iter_games(args.inp):
        if args.max_games and done >= args.max_games:
            break
        try:
            samples = parse_one_game(game_lines, winner_only=winner_only,
                                     all_players=args.all_players)
            if not samples:
                skipped += 1
            else:
                for obs, mask, act in samples:
                    obs_list.append(obs)
                    mask_list.append(mask)
                    act_list.append(act)
                done += 1
        except Exception:
            errs += 1

        if (done + errs) % 5000 == 0 and (done + errs) > 0:
            print(f"  {done} games / {len(obs_list)} samples / {errs} errors",
                  flush=True)

    if not obs_list:
        print("No samples extracted!")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        obs=np.array(obs_list,  dtype=np.uint8),
        mask=np.array(mask_list, dtype=np.bool_),
        act=np.array(act_list,   dtype=np.int16),
    )
    print(f"\nGames: {done} parsed, {skipped} skipped, {errs} errors")
    print(f"Samples: {len(obs_list)} -> {args.out}")

    from collections import Counter
    ac = np.array(act_list)
    print("Action distribution (top 10):")
    for act, cnt in Counter(ac.tolist()).most_common(10):
        pct = 100 * cnt / len(act_list)
        name = "Pass" if act==0 else "Hu" if act==1 else f"Play" if act<36 else f"Chi" if act<99 else f"Peng" if act<133 else "Gang+"
        print(f"  act={act:3d} ({name:<5}) {cnt:8d}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
