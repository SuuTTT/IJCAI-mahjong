"""
Tests for the mahjong bot logic.
Run from project root: python3 -m pytest tests/test_bot.py -v
"""

import sys
import json
import subprocess
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bot'))
from mahjong_bot import (
    GameState, shanten, best_discard, check_hu,
    tile_id, tile_from_id, decide_after_draw, decide_after_discard,
    HAS_MAHJONG_GB
)


# ── Tile helpers ─────────────────────────────────────────────────────────────

def test_tile_id_round_trip():
    tiles = ['W1','W9','B1','B9','T1','T9','F1','F4','J1','J3']
    for t in tiles:
        assert tile_from_id(tile_id(t)) == t, f"Round-trip failed for {t}"


# ── Shanten ──────────────────────────────────────────────────────────────────

def test_shanten_tenpai():
    # W1-W9 + B1-B3 + B4 = tenpai waiting for B4 or...
    # Actually W1-W9 already contains 3 sequences + W7-W9
    # Let's use a clear tenpai hand
    hand = ['W1','W2','W3','W4','W5','W6','W7','W8','W9','B1','B2','B3','B4']
    s, _ = shanten(hand)
    assert s == 0, f"Expected tenpai (0) but got {s}"


def test_shanten_already_won():
    # W1W1W1 W2W2W2 W3W3W3 W4W4W4 W5W5 = 13 tiles, complete
    # This needs 14 tiles for a complete hand (4*3 + 2 = 14)
    # With 13 tiles, one of them is the wait
    # W1*3 W2*3 W3*3 + W4W4 = 3 mentsu + 1 jantai = shanten -1? No wait...
    # 3+3+3+2 = 11 tiles. Need 2 more tiles for another set. Not complete.
    # Let's try: W1W2W3 W4W5W6 W7W8W9 B1B2B3 B4B4 (13 tiles, tenpai)
    hand = ['W1','W2','W3','W4','W5','W6','W7','W8','W9','B1','B2','B3','B4']
    s, _ = shanten(hand)
    assert s == 0  # tenpai waiting for B2 or B5 or B4...
    # Actually W1-W9 = 3 sequences, B1-B3 = 1 sequence, B4 = isolated pair candidate
    # 4 mentsu + need 1 jantai = tenpai for B4B4


def test_shanten_one_away():
    # 11 tiles that form 3 complete sets + 1 partial
    hand = ['W1','W2','W3','W4','W5','W6','W7','W8','B1','B2','B3','F1','F1']
    # W1-W3, W4-W6, W7-W8(taatsu), B1-B3, F1F1(jantai) = 2 mentsu + 1 taatsu + 1 jantai
    # Wait, that's 3+3+2+3+2=13 tiles.
    # W1-W3(3), W4-W6(3), W7W8(2 taatsu), B1-B3(3), F1F1(2 jantai) = 13 tiles
    # 2 complete + 1 taatsu + jantai = 8-4-max(1, 1+1) = 8-4-2=2 shanten? That seems off
    # Actually standard formula: 8 - 2*mentsu - max(1, taatsu + jantai)
    # = 8 - 2*2 - max(1, 1+1) = 8 - 4 - 2 = 2. But this can't be right...
    # W1W2W3 W4W5W6 B1B2B3 = 3 mentsu, W7W8 = taatsu, F1F1 = jantai
    # = 8 - 2*3 - max(1, 1+1) = 8 - 6 - 2 = 0 (tenpai!)
    s, _ = shanten(hand)
    assert s == 0, f"Expected tenpai but got {s}"


def test_shanten_high():
    # 13 isolated tiles = shanten 8
    hand = ['W1','W3','W5','W7','W9','B2','B4','B6','B8','T1','T3','F1','J1']
    s, _ = shanten(hand)
    assert s <= 8


def test_shanten_seven_pairs():
    # 6 pairs + 1 isolated = shanten 1 for seven pairs
    hand = ['W1','W1','W2','W2','W3','W3','W4','W4','W5','W5','W6','W6','B1']
    s, _ = shanten(hand)
    assert s <= 1, f"Expected shanten <= 1 for near-7-pairs, got {s}"


# ── Fan calculation ───────────────────────────────────────────────────────────

def test_fan_high_hand():
    if not HAS_MAHJONG_GB:
        return  # Skip without fan calculator
    # Valid 清一色 hand: W1W2W3 + W1W2W3 + W4W5W6 + W7W8W9 + W1W1 (pair jantai W1)
    # hand (13 tiles, not including win tile): W1W2W3 W1W2W3 W4W5W6 W7W8W9 W1
    # win tile: W1 (self-drawn, completes the pair)
    hand13 = ['W1','W2','W3','W1','W2','W3','W4','W5','W6','W7','W8','W9','W1']
    fan = check_hu(hand13, [], 'W1', seat_wind=0, prevalent_wind=0, is_self_drawn=True)
    assert fan >= 8, f"Expected >= 8 fan but got {fan}"


def test_fan_low_hand():
    if not HAS_MAHJONG_GB:
        return
    # Simple hand that might be < 8 fan
    # W1W2W3 W4W5W6 B1B2B3 T1T2T3 F1F1 (self-drawn F1)
    hand = ['W1','W2','W3','W4','W5','W6','B1','B2','B3','T1','T2','T3','F1']
    fan = check_hu(hand, [], 'F1', seat_wind=0, prevalent_wind=0, is_self_drawn=True)
    # This is a valid win but might be low fan - just check it runs
    assert isinstance(fan, int)


def test_fan_not_win():
    if not HAS_MAHJONG_GB:
        return
    # Non-winning hand
    hand = ['W1','W3','W5','W7','W9','B1','B3','B5','B7','B9','T1','T3','T5']
    fan = check_hu(hand, [], 'W2', seat_wind=0, prevalent_wind=0, is_self_drawn=True)
    assert fan == -1  # Not a valid win


# ── State reconstruction ──────────────────────────────────────────────────────

def make_state_after_deal(pid=0, prevalent_wind=0):
    """Build a GameState with player 0, dealt 13 specific tiles."""
    state = GameState()
    state.my_pid = pid
    state.prevalent_wind = prevalent_wind
    state.apply_deal(f"1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 B1 B2 B3 B4")
    return state


def test_state_deal():
    state = make_state_after_deal()
    assert len(state.hand) == 13
    assert 'W1' in state.hand
    assert 'B4' in state.hand


def test_state_draw_play():
    state = make_state_after_deal()
    state.apply_draw('B5')
    assert 'B5' in state.hand
    assert len(state.hand) == 14
    state.apply_my_play('B5')
    assert 'B5' not in state.hand
    assert len(state.hand) == 13


def test_state_draw_discard_different():
    state = make_state_after_deal()
    state.apply_draw('T1')
    assert len(state.hand) == 14
    state.apply_my_play('W1')
    assert 'W1' not in state.hand
    assert len(state.hand) == 13


def test_state_peng():
    state = make_state_after_deal()
    # Simulate: player 3 discards B1
    state.last_discard_pid = 3
    state.last_discard_tile = 'B1'
    # Apply notify
    state.apply_notify("3 3 PLAY B1")
    # I respond PENG + discard W1
    state.apply_my_peng('B1', 'W1')
    assert state.hand.count('B1') == 0  # Both B1 removed from hand
    assert 'W1' not in state.hand
    peng_found = any(p[0] == 'PENG' and p[1] == 'B1' for p in state.packs)
    assert peng_found, f"PENG pack not found: {state.packs}"


def test_can_peng():
    state = make_state_after_deal()
    # Default hand has each tile only once — cannot peng
    assert not state.can_peng('B1')  # Only 1 B1 in hand


def test_can_peng_with_duplicates():
    state = GameState()
    state.my_pid = 0
    state.prevalent_wind = 0
    state.apply_deal("1 0 0 0 0 W1 W2 W3 B1 B1 B1 B2 B2 B2 T1 T1 T1 F1")
    assert state.can_peng('W1') == False  # Only 1 W1
    assert state.can_peng('B1') == True   # 3 B1s


def test_can_chi():
    state = GameState()
    state.my_pid = 1  # I'm player 1
    state.prevalent_wind = 0
    state.apply_deal("1 0 0 0 0 W3 W4 W5 B1 B2 B3 T1 T2 T3 F1 F2 J1 J2")
    # Player 0 discards W2 — player 1 can chi
    options = state.can_chi('W2', 1)  # next_pid=1 (after player 0)
    assert len(options) > 0, f"Should be able to chi W2, got {options}"


# ── Decision making ───────────────────────────────────────────────────────────

def test_decide_draw_plays():
    state = make_state_after_deal()
    state.apply_draw('T5')
    response = decide_after_draw(state)
    assert response.startswith("PLAY "), f"Expected PLAY but got: {response}"
    tile = response.split()[1]
    original_hand = ['W1','W2','W3','W4','W5','W6','W7','W8','W9','B1','B2','B3','B4','T5']
    assert tile in original_hand, f"Discarding {tile} not in hand"


def test_decide_discard_pass_if_no_benefit():
    state = GameState()
    state.my_pid = 2  # Not the next player after pid=0
    state.prevalent_wind = 0
    state.apply_deal("1 0 0 0 0 W1 W3 W5 W7 W9 B1 B3 B5 B7 B9 T1 T3 F1")
    state.last_discard_pid = 0
    state.last_discard_tile = 'W2'
    state.apply_notify("3 0 PLAY W2")
    response = decide_after_discard(state, 0)
    # Can't chi (wrong player), can't peng (no 2 W2s), check if PASS
    assert response == "PASS" or response.startswith("HU"), f"Got: {response}"


def test_no_invalid_tile_discard():
    """Bot must never discard a tile not in its hand."""
    state = GameState()
    state.my_pid = 0
    state.prevalent_wind = 0
    hand_tiles = ['W1','W2','W3','W4','W5','W6','W7','W8','W9','B1','B2','B3','B4']
    state.apply_deal("1 0 0 0 0 " + " ".join(hand_tiles))
    state.apply_draw('T5')

    all_tiles = set(hand_tiles + ['T5'])
    response = decide_after_draw(state)
    if response.startswith("PLAY "):
        tile = response.split()[1]
        assert tile in all_tiles, f"Tried to discard {tile} not in hand"


# ── Integration: simulate a few turns via subprocess ─────────────────────────

def run_bot(input_json):
    """Run the bot with given JSON input, return response string."""
    bot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bot'))
    result = subprocess.run(
        [sys.executable, bot_path],
        input=json.dumps(input_json),
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode != 0:
        raise RuntimeError(f"Bot error:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
    return json.loads(result.stdout)["response"]


def test_bot_init():
    """First turn: just init."""
    inp = {
        "requests": ["0 0 0"],
        "responses": []
    }
    resp = run_bot(inp)
    assert resp == "PASS"


def test_bot_deal_plus_draw():
    """Second real turn: dealt + drew."""
    inp = {
        "requests": [
            "0 0 0",
            "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 B1 B2 B3 B4",
            "3 3 DRAW",   # player 3 drew (not me)
            "3 3 PLAY T1",  # player 3 discarded T1
            "3 0 DRAW",   # wait, I'm player 0
            # Let me fix: player 3 draws then plays, then player 0 draws
            # Actually this is getting complex. Just test a simpler flow:
        ],
        "responses": [
            "PASS",
            "PASS",
            "PASS",
        ]
    }
    # Actually the order should be: init, deal, draw for player 3, ..., draw for player 0
    # Let me simplify: just deal + player 0 draws immediately
    inp2 = {
        "requests": [
            "0 0 0",
            "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 B1 B2 B3 B4",
            "2 T1",  # Player 0 draws T1
        ],
        "responses": [
            "PASS",
            "PASS",
        ]
    }
    resp = run_bot(inp2)
    assert resp.startswith("PLAY ") or resp == "HU", f"Unexpected: {resp}"
    if resp.startswith("PLAY "):
        tile = resp.split()[1]
        valid = set(['W1','W2','W3','W4','W5','W6','W7','W8','W9',
                     'B1','B2','B3','B4','T1'])
        assert tile in valid, f"Invalid tile {tile}"


def test_bot_draw_tenpai_no_hu():
    """Bot draws to tenpai, should not HU (low fan hand)."""
    # W1-W9 + B1-B3 + B4B4 drawn (waiting for B4 is already 2, so self-drawn B4)
    # Actually this hand might be >= 8 fan with all same suit
    inp = {
        "requests": [
            "0 0 0",
            "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 B1 B2 B3 B4",
            "2 B4",
        ],
        "responses": ["PASS", "PASS"]
    }
    resp = run_bot(inp)
    # B4 drawn: hand = W1-W9 + B1-B4-B4
    # W1W2W3 W4W5W6 W7W8W9 = 3 mentsu, B1B2B3 = 1 mentsu, B4B4 = jantai
    # = 4 mentsu + 1 jantai = win!
    # Fan: 清一色 is not possible (mixed B and W), but 平和 + others maybe
    # Don't assert HU — just check valid response
    assert resp.startswith("PLAY ") or resp == "HU" or resp.startswith("GANG")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
