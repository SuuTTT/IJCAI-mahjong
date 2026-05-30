"""
Stress test: run simulated game sequences through the C++ bot
and verify no illegal responses are produced.

Protocol:
  requests[0..n-1] are historical; responses[0..n-1] match them.
  requests[n] is the CURRENT pending request (no response yet).
  We always maintain len(requests) == len(responses) + 1 on bot calls.
"""

import subprocess
import json
import random
import sys
import os

BOT = os.path.join(os.path.dirname(__file__), '..', 'bot', 'bot_submit_test')
SUITS = ['W', 'B', 'T']
HONORS = ['F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3']
FULL_DECK = ([f"{s}{n}" for s in SUITS for n in range(1, 10)] * 4 + HONORS * 4)


def run_bot(requests, responses):
    """Call bot; requests must have exactly one more element than responses."""
    assert len(requests) == len(responses) + 1
    inp = {"requests": list(requests), "responses": list(responses)}
    result = subprocess.run([BOT], input=json.dumps(inp), capture_output=True,
                            text=True, timeout=3)
    if result.returncode != 0:
        raise RuntimeError(f"Bot crash: {result.stderr[:300]}")
    return json.loads(result.stdout)["response"]


def bot_exchange(requests, responses, new_req):
    """
    Append new_req to requests, call bot, store response in responses.
    Returns the bot's response.
    """
    requests.append(new_req)
    resp = run_bot(requests, responses)
    responses.append(resp)
    return resp


def run_game_simulation(seed):
    random.seed(seed)
    deck = FULL_DECK[:]
    random.shuffle(deck)

    hand = deck[:13]
    deck = deck[13:]
    for _ in range(3):
        deck = deck[13:]

    requests  = []
    responses = []
    errors    = []
    turns     = 0

    # Request 0: init
    resp = bot_exchange(requests, responses, "0 0 0")
    if resp != "PASS":
        errors.append(f"Init: expected PASS, got {resp}")

    # Request 1: deal
    deal_req = "1 0 0 0 0 " + " ".join(hand)
    resp = bot_exchange(requests, responses, deal_req)
    if resp != "PASS":
        errors.append(f"Deal: expected PASS, got {resp}")

    current_hand = list(hand)

    for turn in range(40):
        if not deck:
            break

        # --- Player 0 draws ---
        drawn = deck.pop(0)
        current_hand.append(drawn)
        resp = bot_exchange(requests, responses, f"2 {drawn}")

        if resp == "HU":
            break
        elif resp.startswith("PLAY "):
            tile = resp.split()[1]
            if tile not in current_hand:
                errors.append(f"T{turn}: PLAY {tile} not in hand")
                break
            current_hand.remove(tile)
            turns += 1
            # Notify own PLAY
            resp2 = bot_exchange(requests, responses, f"3 0 PLAY {tile}")
            if resp2 != "PASS":
                errors.append(f"T{turn}: self PLAY notify: expected PASS got {resp2}")
        elif resp.startswith("GANG "):
            tile = resp.split()[1]
            c = current_hand.count(tile)
            if c < 4:
                errors.append(f"T{turn}: GANG {tile} but count={c}")
                break
            for _ in range(4): current_hand.remove(tile)
            turns += 1
            bot_exchange(requests, responses, f"3 0 GANG")
        elif resp.startswith("BUGANG "):
            tile = resp.split()[1]
            if tile not in current_hand:
                errors.append(f"T{turn}: BUGANG {tile} not in hand")
                break
            current_hand.remove(tile)
            turns += 1
            bot_exchange(requests, responses, f"3 0 BUGANG {tile}")
        else:
            errors.append(f"T{turn}: bad draw response: '{resp}'")
            break

        if errors:
            break

        # --- Simplified: player 1 draws and plays ---
        if not deck:
            break
        other_tile = deck.pop(0)
        bot_exchange(requests, responses, "3 1 DRAW")
        if not deck:
            break
        play_tile = deck.pop(0)
        resp4 = bot_exchange(requests, responses, f"3 1 PLAY {play_tile}")
        ok = any(resp4.startswith(p) for p in ("PASS","HU","PENG ","GANG","CHI "))
        if not ok:
            errors.append(f"T{turn}: bad response to other PLAY: '{resp4}'")
            break

    return errors, turns


def main():
    n_games = 200
    total_errors = 0
    total_turns  = 0

    print(f"Running {n_games} game simulations...")
    for seed in range(n_games):
        try:
            errors, turns = run_game_simulation(seed)
        except Exception as e:
            errors = [f"Exception: {e}"]
            turns  = 0
        total_turns += turns
        if errors:
            total_errors += len(errors)
            if total_errors <= 15:
                print(f"  Seed {seed}: {errors}")

    print(f"\nResults: {n_games} games, {total_turns} draw turns")
    if total_errors == 0:
        print("All games passed! No illegal moves detected.")
        return 0
    else:
        print(f"FAILED: {total_errors} errors found.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
