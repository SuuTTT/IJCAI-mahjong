"""Acceptance tests for the MCR (国标麻将) engine.

Run on the box:
    cd /root/ludus && source /root/venv/bin/activate && \
    PYTHONPATH=/root/ludus JAX_PLATFORMS=cpu python -m pytest tests/test_mahjong.py -q
"""

from collections import Counter

import pytest

from mahjong.engine import (
    play_game, build_wall, all_tile_codes, meld_tiles, _Game,
)
from mahjong.bots import RandomLegalBot


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def full_multiset():
    tiles = []
    for code in all_tile_codes():
        tiles.extend([code] * 4)
    return tiles


def count_all_tiles(result):
    f = result["_final"]
    c = Counter()
    for h in f["hands"]:
        c.update(h)
    for ms in f["melds"]:
        for m in ms:
            c.update(meld_tiles(tuple(m)))
    for d in f["discards"]:
        c.update(d)
    for w in f["wall_remaining"]:
        c.update(w)
    return c


class ScriptedHUBot:
    """Declares HU on its very first draw; otherwise passes/discards."""

    def __init__(self):
        self.hand = []

    def respond(self, request):
        parts = request.split()
        if not parts:
            return "PASS"
        if parts[0] == "1":
            self.hand = list(parts[5:18])
            return "PASS"
        if parts[0] == "2":
            return "HU"
        return "PASS"


class CannedAgent:
    """Returns a fixed reply to any claimable-discard broadcast."""

    def __init__(self, reply="PASS"):
        self.reply = reply

    def respond(self, request):
        if request.startswith("3") and "PLAY" in request:
            return self.reply
        return "PASS"


QINGLONG_HAND13 = ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9",
                   "T1", "T2", "T3", "B9"]  # + draw B9 -> 清龙


def qinglong_wall():
    """Seat 0's segment yields a guaranteed 清龙 on its first draw (B9)."""
    head = QINGLONG_HAND13 + ["B9"]  # 14 tiles; index 13 (first draw) == B9
    pool = full_multiset()
    for t in head:
        pool.remove(t)
    seg0 = head + pool[:20]
    rest = pool[20:]
    wall = seg0 + rest
    assert len(wall) == 136
    return wall


# ---------------------------------------------------------------------------
# Test 1: runs & conserves tiles
# ---------------------------------------------------------------------------
def test_runs_and_conserves_tiles():
    expected = Counter({c: 4 for c in all_tile_codes()})
    for seed in range(20):
        agents = [RandomLegalBot(seed=seed * 10 + i) for i in range(4)]
        result = play_game(agents, seed=seed)
        assert result["ending"] in ("zimo", "rong", "draw")
        c = count_all_tiles(result)
        assert sum(c.values()) == 136, f"seed {seed}: {sum(c.values())} tiles"
        assert c == expected, f"seed {seed}: tile multiset mismatch"


# ---------------------------------------------------------------------------
# Test 2: exhaustive draw is reachable, scores all zero
# ---------------------------------------------------------------------------
def test_exhaustive_draw():
    saw_draw = False
    for seed in range(30):
        agents = [RandomLegalBot(seed=seed * 3 + i) for i in range(4)]
        result = play_game(agents, seed=seed)
        if result["ending"] == "draw":
            saw_draw = True
            assert result["scores"] == {0: 0, 1: 0, 2: 0, 3: 0}
            assert sum(result["scores"].values()) == 0
            assert result["winner"] is None
    assert saw_draw, "no exhaustive draw in 30 random games"


# ---------------------------------------------------------------------------
# Test 3: fan / win validation
# ---------------------------------------------------------------------------
def test_zimo_qinglong_win():
    agents = [ScriptedHUBot()] + [RandomLegalBot(seed=i) for i in range(1, 4)]
    result = play_game(agents, wall=qinglong_wall(), quan=0, seed=0)
    assert result["ending"] == "zimo"
    assert result["winner"] == 0
    assert result["fan"] >= 8
    assert sum(result["scores"].values()) == 0
    assert result["scores"][0] > 0
    # zimo: winner +3*(8+F); each other -(8+F)
    f = result["fan"]
    assert result["scores"][0] == 3 * (8 + f)
    for s in (1, 2, 3):
        assert result["scores"][s] == -(8 + f)


def test_low_fan_hu_rejected():
    """A structurally-complete but <8-fan hand is NOT a valid win."""
    game = _Game([CannedAgent() for _ in range(4)], build_wall(0), 0, 0)
    # >=8 fan: 清龙 completes on B9
    game.hands[0] = QINGLONG_HAND13 + ["B9"]
    assert game._try_win(0, "B9", is_self=True, about_kong=False) >= 8
    # <8 fan: 3 chows + a triplet + honor pair == 6 fan self-draw
    game.hands[1] = ["W2", "W3", "W4", "W5", "W6", "W7",
                     "T2", "T3", "T4", "B5", "B5", "F2", "F2", "B5"]
    assert game._try_win(1, "B5", is_self=True, about_kong=False) is None
    # not a win at all
    game.hands[2] = ["W1", "T5", "B9", "F1", "J3", "W4", "T2",
                     "B6", "F3", "J1", "W8", "T7", "B3", "F4"]
    assert game._try_win(2, "F4", is_self=True, about_kong=False) is None


# ---------------------------------------------------------------------------
# Test 4: claim priority
# ---------------------------------------------------------------------------
def test_peng_beats_chi():
    agents = [CannedAgent("PASS"),          # seat 0 discarder (not queried)
              CannedAgent("CHI W5 J1"),     # seat 1 (next) would CHI
              CannedAgent("PENG J2"),       # seat 2 wants PENG (higher priority)
              CannedAgent("PASS")]
    game = _Game(agents, build_wall(0), 0, 0)
    game.hands[1] = ["W4", "W6", "J1", "J1"]
    game.hands[2] = ["W5", "W5", "J2", "J2"]
    game.hands[3] = ["B1", "B2"]
    game.discards[0] = ["W5"]
    out = game._resolve(0, "W5", "3 0 PLAY W5")
    assert out[0] == "chain"
    assert out[1][0] == 2, "PENG (seat 2) must win over CHI (seat 1)"
    assert game.melds[2] and game.melds[2][0][:2] == ("PENG", "W5")
    assert game.melds[1] == [], "seat 1 must not have chi'd"


def test_hu_beats_peng():
    agents = [CannedAgent("PASS"),
              CannedAgent("HU"),            # seat 1 can rong on B9 (清龙)
              CannedAgent("PENG J2"),       # seat 2 wants PENG
              CannedAgent("PASS")]
    game = _Game(agents, build_wall(0), 0, 0)
    game.hands[1] = list(QINGLONG_HAND13)   # 13 tiles, rong on B9 -> 清龙
    game.hands[2] = ["B9", "B9", "J2", "J2"]
    game.discards[0] = ["B9"]
    out = game._resolve(0, "B9", "3 0 PLAY B9")
    assert out[0] == "win"
    assert out[1] == 1, "HU must win over PENG"
    assert out[2] >= 8
    assert game.melds[2] == [], "seat 2 must not have punged"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
