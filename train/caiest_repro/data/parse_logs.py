"""
parse_logs.py — convert game JSONL logs → (obs, valid_mask, action) .npz dataset.

Each game log contains alternating judge-broadcast / bot-response pairs.
We replay the game, feed each turn into FeatureAgent, and record
(observation, legal_action_mask, chosen_action) for the winning player's turns.

Usage:
    python3 data/parse_logs.py \
        --inp data/raw/selfplay.jsonl \
        --out data/processed/selfplay.npz \
        [--winner-only]    # only use winning player's decisions
        [--all-players]    # use all players' decisions (more data, lower quality)
"""

import argparse
import json
import os
import sys
import numpy as np
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.feature_agent import FeatureAgent, ACT, ACT_DIM, OBS_DIM, TILE_INDEX

# ── helpers ────────────────────────────────────────────────────────────────────

def botzone_to_engine(req: str) -> str:
    """Convert raw Botzone request (0/1/2/3 ...) to botzone_engine format."""
    parts = req.strip().split()
    if not parts:
        return ""
    rtype = parts[0]
    if rtype == "0":
        return f"Wind {parts[2]}"   # parts[1]=playerID (ignored), parts[2]=quan
    if rtype == "1":
        tiles = parts[5:]           # skip "1 f0 f1 f2 f3"
        return "Deal " + " ".join(tiles)
    if rtype == "2":
        return f"Draw {parts[1]}"
    if rtype == "3":
        pid, action = parts[1], parts[2]
        rest = " ".join(parts[3:])
        mapping = {
            "DRAW":   f"Player {pid} Draw",
            "PLAY":   f"Player {pid} Play {rest}",
            "PENG":   f"Player {pid} Peng",
            "CHI":    f"Player {pid} Chi {parts[3] if len(parts)>3 else rest}",
            "GANG":   f"Player {pid} Gang",
            "BUGANG": f"Player {pid} BuGang {rest}",
        }
        return mapping.get(action, f"Player {pid} {action} {rest}")
    return ""


def response_to_action(resp: str) -> int:
    """Convert a bot's response string to an action index."""
    parts = resp.strip().split()
    if not parts:
        return ACT["Pass"]
    w = parts[0].upper()
    if w == "PASS":                              return ACT["Pass"]
    if w == "HU":                                return ACT["Hu"]
    if w == "PLAY" and len(parts) > 1:
        t = parts[1]
        return ACT["Play"] + TILE_INDEX.get(t, 0)
    if w == "PENG" and len(parts) > 1:
        # PENG <discard_after> — the action is PENG + tile of what was penged
        # We don't have that here; store as generic Peng with tile_idx=0
        return ACT["Peng"]   # imprecise but OK for this pass
    if w == "GANG":
        return ACT["Gang"]
    if w == "CHI" and len(parts) >= 3:
        # CHI mid_tile discard_tile
        mid_tile = parts[1]
        suit, mid_n = mid_tile[0], int(mid_tile[1])
        # discard tile could come from request context, not available here
        # approximate: use position 1 (middle of sequence)
        from data.feature_agent import chi_action_idx
        return chi_action_idx(suit, mid_n, mid_n)
    if w == "GANG":                              return ACT["Gang"]
    if w == "BUGANG" and len(parts) > 1:
        t = parts[1]
        return ACT["BuGang"] + TILE_INDEX.get(t, 0)
    return ACT["Pass"]


def parse_one_game(game: dict, winner_only: bool = True) -> List[Tuple]:
    """
    Returns list of (obs[240], valid_mask[235], action_idx) tuples.
    """
    log   = game.get("log", [])
    quan  = game.get("quan", 0)
    scores = game.get("scores", [0,0,0,0])
    winner = -1
    for i, s in enumerate(scores):
        if s > 0:
            winner = i
            break

    if winner_only and winner == -1:
        return []   # 荒牌 — skip

    agents = [FeatureAgent(i) for i in range(4)]
    samples = []

    # log alternates: [broadcast_0, responses_0, broadcast_1, responses_1, ...]
    for round_idx in range(0, len(log) - 1, 2):
        broadcast = log[round_idx]      # dict: pid -> request_str
        responses = log[round_idx + 1]  # dict: pid -> {verdict, response}

        for pid in range(4):
            req_str  = broadcast.get(str(pid), "")
            resp_obj = responses.get(str(pid), {})
            resp_str = resp_obj.get("response", "PASS") if isinstance(resp_obj, dict) else "PASS"

            eng_req = botzone_to_engine(req_str)
            if not eng_req:
                continue

            obs, valid = agents[pid].update(eng_req)

            # Only record draw decisions (type "2") or after-discard responses
            req_parts = req_str.strip().split()
            is_draw    = req_parts[0] == "2"
            is_respond = (req_parts[0] == "3" and
                          len(req_parts) > 2 and
                          req_parts[2] == "PLAY" and
                          int(req_parts[1]) != pid)

            if (is_draw or is_respond) and (not winner_only or pid == winner):
                act_idx = response_to_action(resp_str)
                valid_mask = np.zeros(ACT_DIM, dtype=np.bool_)
                for v in valid:
                    if 0 <= v < ACT_DIM:
                        valid_mask[v] = True

                if act_idx in valid and len(valid) > 1:
                    samples.append((obs.copy(), valid_mask, act_idx))

    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inp",   required=True)
    p.add_argument("--out",   required=True)
    p.add_argument("--winner-only",  action="store_true", default=True)
    p.add_argument("--all-players",  action="store_true")
    args = p.parse_args()

    winner_only = not args.all_players

    obs_list, mask_list, act_list = [], [], []
    games = errors = skipped = 0

    with open(args.inp) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                game = json.loads(line)
                if "error" in game:
                    errors += 1
                    continue
                samples = parse_one_game(game, winner_only=winner_only)
                if not samples:
                    skipped += 1
                    continue
                for obs, mask, act in samples:
                    obs_list.append(obs)
                    mask_list.append(mask)
                    act_list.append(act)
                games += 1
            except Exception as e:
                errors += 1

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
    print(f"Games: {games} parsed, {skipped} skipped (no winner), {errors} errors")
    print(f"Samples: {len(obs_list)} total -> {args.out}")
    print(f"Action distribution (top 10):")
    act_arr = np.array(act_list)
    from collections import Counter
    for act, cnt in Counter(act_arr.tolist()).most_common(10):
        print(f"  act={act:3d}  count={cnt:6d}")


if __name__ == "__main__":
    main()
