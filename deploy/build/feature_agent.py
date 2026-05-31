"""
feature_agent.py — faithful port of FeatureAgent2Adapted from Mahjong-LLM/sample.py.

Observation vector: 240 uint8 values
  [0]      PREVALENT_WIND (0-3)
  [1]      SEAT_WIND (0-3)
  [2:36]   UNSHOWN — 4 minus how many of each tile have been seen (34 tiles)
  [36:50]  HAND — up to 14 tile indices (255=empty slot)
  [50:60]  WALL — first 10 known tiles from my private wall (255=empty)
  [60:240] PLAYERS — 4 × 45 bytes
    [p*45 : p*45+29]   HISTORY — up to 29 action tokens (255=empty)
    [p*45+29 : p*45+45] MELDS — 4 melds × 4 bytes (tile indices, 255=empty)

Action indices (235 total):
  0        Pass
  1        Hu
  2-35     Play tile_i
  36-98    Chi  (63 = 7 mid-tiles × 3 positions × 3 suits)
  99-132   Peng tile_i
  133-166  Gang tile_i (from discard)
  167-200  AnGang tile_i (concealed)
  201-234  BuGang tile_i
"""

import numpy as np
from typing import List, Optional, Tuple

try:
    from MahjongGB import MahjongFanCalculator
    HAS_FAN = True
except ImportError:
    HAS_FAN = False

# ── Tile ordering (matches sample.py exactly) ─────────────────────────────────
# W=Characters, T=Bamboo, B=Dots, F=Winds, J=Dragons
# NOTE: order is W/T/B, not W/B/T as in our bot!

TILE_LIST = (
    [f"W{i+1}" for i in range(9)] +
    [f"T{i+1}" for i in range(9)] +
    [f"B{i+1}" for i in range(9)] +
    [f"F{i+1}" for i in range(4)] +
    [f"J{i+1}" for i in range(3)]
)
TILE_INDEX = {t: i for i, t in enumerate(TILE_LIST)}
TILE_INDEX["PUBLIC"]    = 34   # sentinel for open gang
TILE_INDEX["CONCEALED"] = 35   # sentinel for closed gang
N_TILES = 34

OBS_DIM  = 240
ACT_DIM  = 235
EMPTY    = 255  # empty slot value

# ── Observation offsets ───────────────────────────────────────────────────────
OBS = {
    "PREVALENT_WIND": 0,
    "SEAT_WIND":      1,
    "UNSHOWN":        2,   # 34 bytes
    "HAND":          36,   # 14 bytes
    "WALL":          50,   # 10 bytes
    "PLAYER_START":  60,
    "PLAYER_LEN":    45,
    "MELD_START":    29,
    "MELD_LEN":       4,
}

# ── Action offsets ────────────────────────────────────────────────────────────
ACT = {
    "Pass":   0,
    "Hu":     1,
    "Play":   2,    # +tile_idx (0-33)
    "Chi":   36,    # +chi_idx (0-62)
    "Peng":  99,    # +tile_idx
    "Gang": 133,    # +tile_idx
    "AnGang":167,   # +tile_idx
    "BuGang":201,   # +tile_idx
}


def chi_action_idx(suit: str, mid_n: int, discard_n: int) -> int:
    """
    Return action index for Chi.
    suit: 'W','T','B'
    mid_n: 1-9 (middle tile number)
    discard_n: 1-9 (which tile in the sequence came from the discard)
    """
    suit_off = {"W": 0, "T": 1, "B": 2}[suit]
    # mid can be 2-8 (7 choices), position of discard: 0=left,1=mid,2=right
    pos = discard_n - (mid_n - 1)  # 0,1,2
    return ACT["Chi"] + suit_off * 21 + (mid_n - 2) * 3 + pos


def decode_chi(chi_idx: int) -> Tuple[str, int, int]:
    """Return (suit, mid_n, discard_n) from a Chi action index."""
    idx = chi_idx - ACT["Chi"]
    suit = ["W", "T", "B"][idx // 21]
    rem  = idx % 21
    mid_n = rem // 3 + 2
    pos   = rem % 3
    discard_n = mid_n - 1 + pos
    return suit, mid_n, discard_n


# ── FeatureAgent ──────────────────────────────────────────────────────────────

class FeatureAgent:
    """
    Tracks full game state for one player and produces observations.
    Mirrors FeatureAgent2Adapted from the official Mahjong-LLM sample.

    Usage:
        agent = FeatureAgent(seat_wind=0)
        obs, valid = agent.update("Wind 1")           # request 0
        obs, valid = agent.update("Deal W1 W2 ...")   # request 1
        obs, valid = agent.update("Draw W5")           # request 2 (my draw)
    """

    def __init__(self, seat_wind: int):
        self.seat = seat_wind
        self.obs   = np.full(OBS_DIM, EMPTY, dtype=np.uint8)
        self.valid: List[int] = [ACT["Pass"]]

        self.prevalent_wind = 0
        self.hand:   List[str]       = []
        self.packs:  List[List]      = [[] for _ in range(4)]  # each: (type,tile,offer)
        self.history: List[List[int]] = [[] for _ in range(4)]
        self.shown   = {t: 0 for t in TILE_LIST}
        self.flower  = 0
        self.wall_counts = [34, 34, 34, 34]  # remaining tiles per player wall
        self.cur_tile: Optional[str] = None
        self.wall_last = False
        self.my_wall_last = False

        self.obs[OBS["SEAT_WIND"]] = seat_wind
        self._update_unshown()

    # ── public API ─────────────────────────────────────────────────────────────

    def update(self, request: str) -> Tuple[np.ndarray, List[int]]:
        """
        Process one request string (the higher-level format from botzone_engine).
        Returns (obs, valid_actions).
        """
        self.valid = []
        parts = request.split()

        if parts[0] == "Wind":
            self.prevalent_wind = int(parts[1])
            self.obs[OBS["PREVALENT_WIND"]] = self.prevalent_wind
            self.valid = [ACT["Pass"]]

        elif parts[0] == "Deal":
            tiles = parts[1:]
            self.hand = [t for t in tiles if t[0] != "H"]
            self.flower = sum(1 for t in tiles if t[0] == "H")
            self._update_hand()
            self._update_wall()
            # Duplicate Mahjong: each player draws from their own 34-tile wall;
            # 13 are dealt, so 21 remain per wall. (Init was 34 — off by 13,
            # which made wall-exhaustion detection 13 draws too late.)
            self.wall_counts = [21, 21, 21, 21]
            self.valid = [ACT["Pass"]]

        elif parts[0] == "Draw":
            tile = parts[1]
            if tile[0] == "H":
                self.flower += 1
                self.valid = [ACT["Pass"]]
            else:
                self.hand.append(tile)
                self._update_hand()
                self.wall_counts[self.seat] -= 1
                self.my_wall_last = (self.wall_counts[(self.seat + 1) % 4] == 0)
                self.valid = [ACT["Hu"]]
                for t in set(self.hand):
                    self.valid.append(ACT["Play"] + TILE_INDEX[t])
                for t in set(self.hand):
                    if self.hand.count(t) == 4 and not self.my_wall_last:
                        self.valid.append(ACT["AnGang"] + TILE_INDEX[t])
                for pk in self.packs[self.seat]:
                    if pk[0] == "PENG" and pk[1] in self.hand and not self.my_wall_last:
                        self.valid.append(ACT["BuGang"] + TILE_INDEX[pk[1]])

        elif parts[0] == "Player":
            pid = int(parts[1])
            action = parts[2]
            rest = parts[3:]

            if action == "Draw":
                self.wall_counts[pid] -= 1
                self.wall_last = (self.wall_counts[(pid + 1) % 4] == 0)
                self.valid = [ACT["Pass"]]

            elif action == "Play":
                tile = rest[0]
                self.cur_tile = tile
                self._history_append(pid, ACT["Play"] + TILE_INDEX[tile])
                self.shown[tile] += 1
                self._update_unshown()
                if pid == self.seat:
                    if tile in self.hand:
                        self.hand.remove(tile)
                    self._update_hand()
                    self.valid = [ACT["Pass"]]
                else:
                    self.wall_last = (self.wall_counts[(pid + 1) % 4] == 0)
                    self.valid = []
                    if not self.wall_last:
                        self.valid.append(ACT["Hu"])
                        if self.hand.count(tile) >= 2:
                            self.valid.append(ACT["Peng"] + TILE_INDEX[tile])
                        if self.hand.count(tile) == 3:
                            self.valid.append(ACT["Gang"] + TILE_INDEX[tile])
                        # Chi: only if I'm next player
                        if (pid + 1) % 4 == self.seat and tile[0] in ("W","T","B"):
                            suit, n = tile[0], int(tile[1])
                            for mid in range(max(2,n-1), min(8,n+1)+1):
                                needed = [f"{suit}{mid+d}" for d in (-1,0,1) if f"{suit}{mid+d}" != tile]
                                if all(t in self.hand for t in needed):
                                    self.valid.append(chi_action_idx(suit, mid, n))
                    self.valid.append(ACT["Pass"])

            elif action == "Peng":
                tile = self.cur_tile
                if tile is None: return self.obs.copy(), self.valid
                self._history_append(pid, ACT["Peng"] + TILE_INDEX.get(tile, 0))
                self.shown[tile] = min(4, self.shown.get(tile, 0) + 3)
                self._update_unshown()
                if pid == self.seat:
                    for _ in range(2):
                        if tile in self.hand: self.hand.remove(tile)
                    self.packs[pid].append(("PENG", tile, 0))
                    self._pack_append(pid)
                    self._update_hand()
                    self.valid = [ACT["Play"] + TILE_INDEX[t] for t in set(self.hand)]

            elif action == "Chi":
                mid_tile = rest[0] if rest else None
                if mid_tile is None: return self.obs.copy(), self.valid
                mid_suit, mid_n = mid_tile[0], int(mid_tile[1])
                discard_n = int(self.cur_tile[1]) if (self.cur_tile and self.cur_tile[0] == mid_suit) else mid_n
                self._history_append(pid, chi_action_idx(mid_suit, mid_n, discard_n))
                if pid == self.seat:
                    for d in (-1, 0, 1):
                        t = f"{mid_suit}{mid_n+d}"
                        if t != self.cur_tile and t in self.hand:
                            self.hand.remove(t)
                    self.packs[pid].append(("CHI", mid_tile, discard_n - (mid_n-1)))
                    self._pack_append(pid)
                    self._update_hand()
                    self.valid = [ACT["Play"] + TILE_INDEX[t] for t in set(self.hand)]

            elif action == "Gang":
                tile = self.cur_tile
                self._history_append(pid, ACT["Gang"] + TILE_INDEX[tile])
                self.shown[tile] = 4
                self._update_unshown()
                if pid == self.seat:
                    for _ in range(3): self.hand.remove(tile)
                    self.packs[pid].append(("GANG", tile, 0))
                    self._pack_append(pid)
                    self._update_hand()
                self.valid = [ACT["Pass"]]

            elif action == "AnGang":
                tile = rest[0] if rest else None
                if tile:
                    self._history_append(pid, ACT["AnGang"] + TILE_INDEX.get(tile, 34))
                    if pid == self.seat:
                        for _ in range(4): self.hand.remove(tile)
                        self.packs[pid].append(("GANG", tile, pid))
                        self._pack_append(pid)
                        self._update_hand()
                self.valid = [ACT["Pass"]]

            elif action == "BuGang":
                tile = rest[0]
                self._history_append(pid, ACT["BuGang"] + TILE_INDEX[tile])
                self.cur_tile = tile
                self.shown[tile] = 4
                self._update_unshown()
                if pid == self.seat:
                    self.hand.remove(tile)
                    for i, pk in enumerate(self.packs[pid]):
                        if pk[0] == "PENG" and pk[1] == tile:
                            self.packs[pid][i] = ("GANG", tile, pk[2])
                            break
                    self._pack_append(pid)
                    self._update_hand()
                    self.valid = [ACT["Pass"]]
                else:
                    self.valid = [ACT["Hu"], ACT["Pass"]]

            elif action in ("Un",):
                # Undo (Chi/Peng that didn't happen)
                self._history_pop(pid)
                if pid == self.seat:
                    prev = parts[3] if len(parts) > 3 else ""
                    if prev == "Chi" and len(parts) > 4:
                        mid_tile = parts[4]
                        mid_suit, mid_n = mid_tile[0], int(mid_tile[1])
                        for d in (-1, 0, 1):
                            t = f"{mid_suit}{mid_n+d}"
                            if t != self.cur_tile:
                                self.hand.append(t)
                        if self.packs[pid]:
                            self.packs[pid].pop()
                            self._pack_pop(pid)
                    elif prev == "Peng":
                        tile = self.cur_tile
                        self.hand.append(tile)
                        self.hand.append(tile)
                        if self.packs[pid]:
                            self.packs[pid].pop()
                            self._pack_pop(pid)
                    self._update_hand()
                self.valid = [ACT["Pass"]]

            else:
                self.valid = [ACT["Pass"]]

        return self.obs.copy(), list(self.valid)

    def can_hu(self, win_tile: str, is_self: bool = False,
               is_kong: bool = False) -> int:
        """
        Returns total fan if this is a valid >=8-fan win, else -1.
        The fan calculator expects `hand` = concealed tiles WITHOUT the win tile.
        For a self-draw the drawn tile is already in self.hand, so remove one copy.
        """
        if not HAS_FAN:
            return -1
        concealed = list(self.hand)
        if win_tile in concealed:
            concealed.remove(win_tile)
        # Physical sanity: concealed + melds*3 + winTile must equal 14
        if len(concealed) + 3 * len(self.packs[self.seat]) + 1 != 14:
            return -1
        # Force every meld to EXPOSED (offer=1). The fan calculator treats
        # offer=0 as concealed and awards 门前清 (+2); since we don't perfectly
        # track concealed kongs, forcing exposed yields a conservative LOWER
        # bound on fan — if it still reaches >=8, the judge (which uses the true
        # offers, never higher) will agree. Prevents wrong-HU / -30 penalties.
        safe_packs = tuple((p[0], p[1], 1) for p in self.packs[self.seat])
        try:
            result = MahjongFanCalculator(
                pack=safe_packs,
                hand=tuple(concealed),
                winTile=win_tile,
                flowerCount=self.flower,
                isSelfDrawn=is_self,
                is4thTile=(self.shown.get(win_tile, 0) + int(is_self)) == 4,
                isAboutKong=is_kong,
                isWallLast=self.wall_last or self.my_wall_last,
                seatWind=self.seat,
                prevalentWind=self.prevalent_wind,
            )
            return sum(c for c, _ in result)
        except Exception:
            return -1

    # ── embedding helpers ─────────────────────────────────────────────────────

    def _update_hand(self):
        base = OBS["HAND"]
        self.obs[base: base + 14] = EMPTY
        for i, t in enumerate(self.hand[:14]):
            self.obs[base + i] = TILE_INDEX[t]

    def _update_wall(self):
        """Wall is not normally visible; just zero it out."""
        base = OBS["WALL"]
        self.obs[base: base + 10] = EMPTY

    def _update_unshown(self):
        base = OBS["UNSHOWN"]
        for i, t in enumerate(TILE_LIST):
            self.obs[base + i] = max(0, 4 - self.shown.get(t, 0))

    def _history_append(self, pid: int, token: int):
        h = self.history[pid]
        if len(h) >= 29:
            return
        h.append(token)
        offset = OBS["PLAYER_START"] + OBS["PLAYER_LEN"] * pid + len(h) - 1
        self.obs[offset] = token % 256

    def _history_pop(self, pid: int):
        h = self.history[pid]
        if not h:
            return
        offset = OBS["PLAYER_START"] + OBS["PLAYER_LEN"] * pid + len(h) - 1
        self.obs[offset] = EMPTY
        h.pop()

    def _pack_append(self, pid: int):
        packs = self.packs[pid]
        l = len(packs) - 1
        if l >= 4:
            return
        pt, tile, _ = packs[-1]
        base = (OBS["PLAYER_START"] + OBS["PLAYER_LEN"] * pid +
                OBS["MELD_START"] + OBS["MELD_LEN"] * l)
        ti = TILE_INDEX.get(tile, EMPTY)
        if pt == "CHI":
            for i in range(-1, 2):
                self.obs[base + i + 1] = ti + i
        elif pt == "PENG":
            self.obs[base: base + 3] = ti
        else:  # GANG
            self.obs[base: base + 4] = ti

    def _pack_pop(self, pid: int):
        l = len(self.packs[pid])
        base = (OBS["PLAYER_START"] + OBS["PLAYER_LEN"] * pid +
                OBS["MELD_START"] + OBS["MELD_LEN"] * l)
        self.obs[base: base + 4] = EMPTY
