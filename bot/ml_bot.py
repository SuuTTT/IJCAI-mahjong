"""
ml_bot.py — Keep Running bot backed by trained MLP policy.

Falls back to heuristic (mahjong_bot.py) if model weights aren't available.

Usage (Keep Running protocol, compatible with local_ai.py):
    MODEL=train/checkpoints/bc_v1_weights.npz python3 bot/ml_bot.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from data.feature_agent import FeatureAgent, ACT, ACT_DIM, TILE_LIST, decode_chi
from mahjong_bot import (
    GameState, decide_after_draw, decide_after_discard,
    decide_after_gang_notify, check_hu,
)

SENTINEL = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"
MODEL_PATH = os.environ.get("MODEL", "train/checkpoints/bc_v1_weights.npz")

# ── Load model (optional) ──────────────────────────────────────────────────────
model = None
if os.path.exists(MODEL_PATH):
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from train.model import NumpyMLP
        model = NumpyMLP(MODEL_PATH)
        print(f"[ml_bot] loaded model from {MODEL_PATH}", file=sys.stderr)
    except Exception as e:
        print(f"[ml_bot] model load failed: {e}", file=sys.stderr)

# ── Game state (persists across turns) ────────────────────────────────────────
state     = GameState()
feat_agent = None        # FeatureAgent, reset each game


def reset_game(seat: int, wind: int):
    global state, feat_agent
    state = GameState()
    state.my_pid = seat
    state.prevalent_wind = wind
    feat_agent = FeatureAgent(seat)


# ── Action decoding ────────────────────────────────────────────────────────────

def action_idx_to_botzone(act_idx: int, valid: list) -> str:
    """Convert action index → Botzone response string."""
    if act_idx == ACT["Pass"]:                 return "PASS"
    if act_idx == ACT["Hu"]:                   return "HU"
    if ACT["Play"] <= act_idx < ACT["Chi"]:
        t = TILE_LIST[act_idx - ACT["Play"]]
        return f"PLAY {t}"
    if ACT["Chi"] <= act_idx < ACT["Peng"]:
        suit, mid_n, discard_n = decode_chi(act_idx)
        # After chi, must discard — pick best from hand simulation
        mid_tile = f"{suit}{mid_n}"
        # Build what-discard via heuristic
        disc = _pick_discard_after_chi(suit, mid_n, discard_n)
        return f"CHI {mid_tile} {disc}" if disc else "PASS"
    if ACT["Peng"] <= act_idx < ACT["Gang"]:
        # After peng, must discard — pick best tile
        tile = TILE_LIST[act_idx - ACT["Peng"]]
        disc = _pick_discard_after_peng(tile)
        return f"PENG {disc}" if disc else "PASS"
    if ACT["Gang"] <= act_idx < ACT["AnGang"]:
        return "GANG"
    if ACT["AnGang"] <= act_idx < ACT["BuGang"]:
        t = TILE_LIST[act_idx - ACT["AnGang"]]
        return f"GANG {t}"
    if act_idx >= ACT["BuGang"]:
        t = TILE_LIST[act_idx - ACT["BuGang"]]
        return f"BUGANG {t}"
    return "PASS"


def _pick_discard_after_peng(penged_tile: str) -> str:
    """After penging, pick best tile to discard from remaining hand."""
    from mahjong_bot import shanten, best_discard
    sim_hand = [t for t in state.hand if t != penged_tile]
    # Remove two copies of penged_tile
    rem = 2
    filtered = []
    for t in state.hand:
        if t == penged_tile and rem > 0:
            rem -= 1
        else:
            filtered.append(t)
    return best_discard(filtered, state.packs) or (filtered[0] if filtered else "")


def _pick_discard_after_chi(suit: str, mid_n: int, discard_n: int) -> str:
    """After chii-ing, pick best tile to discard."""
    from mahjong_bot import best_discard
    sim_hand = list(state.hand)
    discard_tile = f"{suit}{discard_n}"
    for d in (-1, 0, 1):
        t = f"{suit}{mid_n+d}"
        if t != discard_tile and t in sim_hand:
            sim_hand.remove(t)
    return best_discard(sim_hand, state.packs) or (sim_hand[0] if sim_hand else "")


# ── Decision making ────────────────────────────────────────────────────────────

def decide_ml(obs, valid: list) -> int:
    """Use model if available, else return -1 to fall back to heuristic."""
    if model is None or not valid:
        return -1
    import numpy as np
    mask = np.zeros(ACT_DIM, dtype=bool)
    for v in valid:
        if 0 <= v < ACT_DIM:
            mask[v] = True
    if mask.sum() == 0:
        return -1
    return model.best_action(obs, mask)


# ── Protocol handler ───────────────────────────────────────────────────────────

def respond(r: str):
    print(r, flush=True)
    print(SENTINEL, flush=True)


def handle(line: str):
    parts = line.strip().split()
    if not parts:
        respond("PASS")
        return

    rtype = parts[0]

    if rtype == "0":
        seat = int(parts[1])
        wind = int(parts[2])
        reset_game(seat, wind)
        if feat_agent:
            feat_agent.update(f"Wind {wind}")
        respond("PASS")

    elif rtype == "1":
        state.apply_deal(line.strip())
        if feat_agent:
            tiles = " ".join(parts[5:])
            feat_agent.update(f"Deal {tiles}")
        respond("PASS")

    elif rtype == "2":
        tile = parts[1]
        state.apply_draw(tile)

        # ML path
        if feat_agent and model:
            obs, valid = feat_agent.update(f"Draw {tile}")
            act_idx = decide_ml(obs, valid)
            if act_idx >= 0:
                response = action_idx_to_botzone(act_idx, valid)
                # Safety: if ML says HU, verify with fan calculator
                if response == "HU":
                    # hand includes drawn tile; check fan excluding it
                    hand_ex = list(state.hand)
                    if tile in hand_ex:
                        hand_ex.remove(tile)
                    fan = check_hu(hand_ex, state.packs, tile,
                                   seat_wind=state.my_pid,
                                   prevalent_wind=state.prevalent_wind,
                                   is_self_drawn=True,
                                   flower_count=state.flower_count)
                    if fan < 8:
                        response = "PASS"  # override: not safe HU
            else:
                response = decide_after_draw(state)
        else:
            response = decide_after_draw(state)

        # Update state
        if response.startswith("PLAY "):
            state.apply_my_play(response.split()[1])
        elif response.startswith("GANG "):
            state.apply_my_gang(response.split()[1])
        elif response.startswith("BUGANG "):
            state.apply_my_bugang(response.split()[1])

        respond(response)

    elif rtype == "3":
        pid    = int(parts[1])
        action = parts[2]
        tile1  = parts[3] if len(parts) > 3 else None

        # Update feature agent with notification
        if feat_agent:
            if action == "DRAW":
                feat_agent.update(f"Player {pid} Draw")
            elif action == "PLAY":
                feat_agent.update(f"Player {pid} Play {tile1}")
            elif action == "PENG":
                feat_agent.update(f"Player {pid} Peng")
            elif action == "CHI":
                feat_agent.update(f"Player {pid} Chi {tile1}")
            elif action == "GANG":
                feat_agent.update(f"Player {pid} Gang")
            elif action == "BUGANG":
                feat_agent.update(f"Player {pid} BuGang {tile1}")

        if pid == state.my_pid:
            state.apply_notify(line.strip())
            respond("PASS")
            return

        state.apply_notify(line.strip())

        if action == "PLAY":
            # ML response
            if feat_agent and model:
                obs, valid = feat_agent.update(f"Player {pid} Play {tile1}")
                # Override: already updated above, get again after
                act_idx = decide_ml(obs, valid)
                if act_idx >= 0:
                    response = action_idx_to_botzone(act_idx, valid)
                    if response == "HU":
                        fan = check_hu(list(state.hand), state.packs, tile1,
                                       seat_wind=state.my_pid,
                                       prevalent_wind=state.prevalent_wind,
                                       is_self_drawn=False,
                                       flower_count=state.flower_count)
                        if fan < 8:
                            response = "PASS"
                else:
                    response = decide_after_discard(state, pid)
            else:
                response = decide_after_discard(state, pid)

            if response.startswith("PENG "):
                state.apply_my_peng(tile1, response.split()[1])
            elif response.startswith("CHI ") and len(response.split()) >= 3:
                parts_r = response.split()
                state.apply_my_chi(parts_r[1], parts_r[2], tile1)
            elif response == "GANG":
                state.apply_my_meld_gang(tile1)
            respond(response)

        elif action in ("GANG", "BUGANG"):
            if feat_agent and model and action == "BUGANG":
                obs, valid = feat_agent.update(f"Player {pid} BuGang {tile1}")
                act_idx = decide_ml(obs, valid)
                if act_idx >= 0:
                    response = action_idx_to_botzone(act_idx, valid)
                    if response == "HU":
                        fan = check_hu(list(state.hand), state.packs, tile1,
                                       seat_wind=state.my_pid,
                                       prevalent_wind=state.prevalent_wind,
                                       is_self_drawn=False, is_about_kong=True,
                                       flower_count=state.flower_count)
                        if fan < 8:
                            response = "PASS"
                else:
                    response = decide_after_gang_notify(state)
            else:
                response = decide_after_gang_notify(state)
            respond(response)
        else:
            respond("PASS")

    else:
        respond("PASS")


# ── main ───────────────────────────────────────────────────────────────────────

def run():
    try:
        handshake = sys.stdin.readline().strip()
        if handshake != "1" and handshake:
            handle(handshake)
    except EOFError:
        return
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if line:
                handle(line)
        except EOFError:
            break
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            respond("PASS")


if __name__ == "__main__":
    run()
