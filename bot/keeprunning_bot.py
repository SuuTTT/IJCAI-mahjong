"""
keeprunning_bot.py — Keep Running protocol wrapper around mahjong_bot.py logic.

Reads one raw Botzone request line at a time from stdin.
After each response prints >>>BOTZONE_REQUEST_KEEP_RUNNING<<<.
Compatible with local_ai/local_ai.py for live Botzone testing.

Run standalone (for manual testing):
    echo "1" | python3 keeprunning_bot.py
    then type requests like:  0 2 1  /  1 0 0 0 0 W1 ...  /  2 T5
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from mahjong_bot import (
    GameState,
    decide_after_draw,
    decide_after_discard,
    decide_after_gang_notify,
    check_hu,
)

SENTINEL = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"

# ── global game state (persists across turns) ──────────────────────────────────
state = GameState()


def reset():
    global state
    state = GameState()


def respond(r: str):
    print(r, flush=True)
    print(SENTINEL, flush=True)


def handle(line: str):
    parts = line.strip().split()
    if not parts:
        respond("PASS")
        return

    rtype = parts[0]

    # ── 0: init ────────────────────────────────────────────────────────────────
    if rtype == "0":
        reset()
        state.my_pid        = int(parts[1])
        state.prevalent_wind = int(parts[2])
        respond("PASS")

    # ── 1: deal ────────────────────────────────────────────────────────────────
    elif rtype == "1":
        state.apply_deal(line.strip())
        respond("PASS")

    # ── 2: my draw ─────────────────────────────────────────────────────────────
    elif rtype == "2":
        tile = parts[1]
        state.apply_draw(tile)
        response = decide_after_draw(state)

        # Update state to reflect what we just decided
        if response.startswith("PLAY "):
            state.apply_my_play(response.split()[1])
        elif response.startswith("GANG "):
            state.apply_my_gang(response.split()[1])
        elif response.startswith("BUGANG "):
            state.apply_my_bugang(response.split()[1])

        # Translate to Botzone format (decide_* already returns "PLAY W5" etc.)
        respond(response)

    # ── 3: notification ────────────────────────────────────────────────────────
    elif rtype == "3":
        pid    = int(parts[1])
        action = parts[2]
        tile1  = parts[3] if len(parts) > 3 else None
        tile2  = parts[4] if len(parts) > 4 else None

        if pid == state.my_pid:
            # My own action notification — always PASS
            # (but still update state for PENG/CHI/GANG we played)
            if action == "PENG" and tile1:
                pass  # state was already updated when we decided PENG
            elif action == "CHI" and tile1:
                pass
            respond("PASS")
            return

        # Update shared state from the notification
        state.apply_notify(line.strip())

        if action == "PLAY":
            response = decide_after_discard(state, pid)
            # If we decided to PENG/CHI/GANG, update state
            if response.startswith("PENG "):
                discard_after = response.split()[1]
                state.apply_my_peng(tile1, discard_after)
            elif response.startswith("CHI ") and len(response.split()) >= 3:
                mid_tile     = response.split()[1]
                discard_after = response.split()[2]
                state.apply_my_chi(mid_tile, discard_after, tile1)
            elif response == "GANG":
                state.apply_my_meld_gang(tile1)
            respond(response)

        elif action in ("GANG", "BUGANG"):
            response = decide_after_gang_notify(state)
            respond(response)

        else:
            respond("PASS")

    else:
        respond("PASS")


# ── main loop ──────────────────────────────────────────────────────────────────

def run():
    # Startup handshake: local_ai.py writes "1" first
    try:
        handshake = sys.stdin.readline().strip()
        if handshake != "1":
            # Not a keep-running session (e.g. direct test) — push back the line
            # by handling it normally
            if handshake:
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
