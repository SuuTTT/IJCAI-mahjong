"""
IJCAI Mahjong AI Bot - v0.1 Safe Heuristic
Protocol: Botzone JSON I/O for Chinese Standard Mahjong (国标麻将)

Strategy:
  - Only HU when fan calculator confirms >= 8 fan
  - Discard tile that minimizes shanten number
  - PENG/CHI when it reduces shanten by >= 1
  - PASS conservatively if uncertain
"""

import sys
import json

try:
    from MahjongGB import MahjongFanCalculator, RegularShanten, SevenPairsShanten, MahjongShanten
    HAS_MAHJONG_GB = True
except ImportError:
    HAS_MAHJONG_GB = False

# ── Tile helpers ──────────────────────────────────────────────────────────────

SUIT_W = 0
SUIT_B = 1
SUIT_T = 2
SUIT_F = 3  # winds
SUIT_J = 4  # arrows (dragons)

def tile_suit(t):
    return {'W': SUIT_W, 'B': SUIT_B, 'T': SUIT_T, 'F': SUIT_F, 'J': SUIT_J}[t[0]]

def tile_num(t):
    return int(t[1])

def is_number_tile(t):
    return t[0] in ('W', 'B', 'T')

def is_honor_tile(t):
    return t[0] in ('F', 'J')

def is_flower_tile(t):
    return t[0] == 'H'

def same_suit(a, b):
    return a[0] == b[0]

def tile_id(t):
    """Map tile string to integer 0-33 (no flowers)."""
    if t[0] == 'W': return int(t[1]) - 1
    if t[0] == 'B': return 9 + int(t[1]) - 1
    if t[0] == 'T': return 18 + int(t[1]) - 1
    if t[0] == 'F': return 27 + int(t[1]) - 1
    if t[0] == 'J': return 31 + int(t[1]) - 1
    return -1

def tile_from_id(i):
    if i < 9: return f'W{i+1}'
    if i < 18: return f'B{i-9+1}'
    if i < 27: return f'T{i-18+1}'
    if i < 31: return f'F{i-27+1}'
    if i < 34: return f'J{i-31+1}'
    return None

# ── Pure-Python shanten calculator ───────────────────────────────────────────

def _shanten_basic_counts(cnt, n_packs):
    """
    cnt: array of 34 ints (tile counts in hand)
    n_packs: number of already-declared packs (each is 1 complete mentsu)
    Returns minimum shanten for standard form (4 mentsu + 1 jantai).
    -1 = already winning, 0 = tenpai.
    """
    best = [8]

    def dfs(pos, mentsu, taatsu, jantai):
        # Prune: calculate current best possible
        m_total = mentsu + n_packs
        val = 8 - 2 * m_total - max(1, taatsu + (1 if jantai else 0))
        if val >= best[0]:
            return
        if pos == 34:
            best[0] = min(best[0], val)
            return

        # Skip empty tiles
        if cnt[pos] == 0:
            dfs(pos + 1, mentsu, taatsu, jantai)
            return

        # Try use pos as pair (jantai)
        if not jantai and cnt[pos] >= 2:
            cnt[pos] -= 2
            dfs(pos + 1, mentsu, taatsu, True)
            cnt[pos] += 2

        # Try use pos as taatsu pair (not jantai)
        if mentsu + taatsu < 4 and cnt[pos] >= 2:
            cnt[pos] -= 2
            dfs(pos + 1, mentsu, taatsu + 1, jantai)
            cnt[pos] += 2

        # For number tiles: sequential taatsu and mentsu
        if pos < 27:
            suit_base = (pos // 9) * 9
            n = pos - suit_base  # 0..8

            # Kanchan taatsu (n, n+2)
            if n <= 6 and mentsu + taatsu < 4 and cnt[pos + 2] >= 1:
                cnt[pos] -= 1
                cnt[pos + 2] -= 1
                dfs(pos + 1, mentsu, taatsu + 1, jantai)
                cnt[pos] += 1
                cnt[pos + 2] += 1

            # Sequential taatsu (n, n+1)
            if n <= 7 and mentsu + taatsu < 4 and cnt[pos + 1] >= 1:
                cnt[pos] -= 1
                cnt[pos + 1] -= 1
                dfs(pos + 1, mentsu, taatsu + 1, jantai)
                cnt[pos] += 1
                cnt[pos + 1] += 1

            # Complete sequence (n, n+1, n+2)
            if n <= 6 and mentsu + taatsu <= 4 and cnt[pos + 1] >= 1 and cnt[pos + 2] >= 1:
                cnt[pos] -= 1
                cnt[pos + 1] -= 1
                cnt[pos + 2] -= 1
                dfs(pos + 1, mentsu + 1, taatsu, jantai)
                cnt[pos] += 1
                cnt[pos + 1] += 1
                cnt[pos + 2] += 1

        # Complete triplet
        if mentsu + taatsu <= 4 and cnt[pos] >= 3:
            cnt[pos] -= 3
            dfs(pos + 1, mentsu + 1, taatsu, jantai)
            cnt[pos] += 3

        # Skip this tile (isolated)
        dfs(pos + 1, mentsu, taatsu, jantai)

    dfs(0, 0, 0, False)
    return best[0]


def _shanten_seven_pairs(cnt):
    """Shanten for 7-pairs form."""
    pairs = sum(1 for c in cnt if c >= 2)
    unique = sum(1 for c in cnt if c >= 1)
    return 6 - pairs + max(0, 7 - unique)


def shanten(hand_tiles, packs=None):
    """
    Calculate shanten number for the given hand.
    hand_tiles: list of tile strings (not including declared packs)
    packs: list of (type, tile, offer) — for counting n_packs
    Returns (shanten_number, form)
    """
    if HAS_MAHJONG_GB:
        try:
            tup = tuple(hand_tiles)
            s_reg, _ = RegularShanten(tup)
            s_7p, _ = SevenPairsShanten(tup)
            s = min(s_reg, s_7p)
            form = 'regular' if s_reg <= s_7p else 'sevenpairs'
            return s, form
        except Exception:
            pass

    # Pure Python fallback
    n_packs = len(packs) if packs else 0
    cnt = [0] * 34
    for t in hand_tiles:
        tid = tile_id(t)
        if 0 <= tid < 34:
            cnt[tid] += 1

    s_reg = _shanten_basic_counts(cnt[:], n_packs)
    s_7p = _shanten_seven_pairs(cnt)
    s = min(s_reg, s_7p)
    form = 'regular' if s_reg <= s_7p else 'sevenpairs'
    return s, form


def best_discard(hand_tiles, packs=None):
    """
    Return the tile to discard that gives the lowest shanten number.
    """
    if not hand_tiles:
        return None

    best_s = 99
    best_tile = hand_tiles[0]

    for i, t in enumerate(hand_tiles):
        remaining = hand_tiles[:i] + hand_tiles[i+1:]
        s, _ = shanten(remaining, packs)
        if s < best_s:
            best_s = s
            best_tile = t

    return best_tile


# ── Fan / HU validation ───────────────────────────────────────────────────────

def check_hu(hand_tiles, packs, win_tile, seat_wind, prevalent_wind,
             is_self_drawn=False, is_about_kong=False, is_wall_last=False,
             is_4th_tile=False, flower_count=0):
    """
    Returns total fan count if hand can win with >= 8 fan, else -1.
    hand_tiles should NOT include win_tile yet.
    """
    if HAS_MAHJONG_GB:
        try:
            pack_arg = tuple(
                (p[0], p[1], p[2]) for p in packs
            )
            hand_arg = tuple(hand_tiles)
            result = MahjongFanCalculator(
                pack=pack_arg,
                hand=hand_arg,
                winTile=win_tile,
                flowerCount=flower_count,
                isSelfDrawn=is_self_drawn,
                is4thTile=is_4th_tile,
                isAboutKong=is_about_kong,
                isWallLast=is_wall_last,
                seatWind=seat_wind,
                prevalentWind=prevalent_wind
            )
            total = sum(cnt for cnt, _ in result)
            return total
        except Exception:
            return -1
    # Without fan calculator, don't call HU (too risky)
    return -1


# ── Game state ────────────────────────────────────────────────────────────────

class GameState:
    def __init__(self):
        self.my_pid = 0
        self.prevalent_wind = 0
        self.hand = []        # tiles in hand (strings)
        self.packs = []       # [(type, tile, offer), ...]
                              #   type: "CHI"/"PENG"/"GANG"
                              #   offer: seat of tile provider (0-3)
        self.flower_count = 0
        self.last_draw = None  # tile just drawn (before we respond)
        self.last_discard_pid = None  # who just discarded
        self.last_discard_tile = None  # what they discarded
        self.last_bugang = False
        self.wall_empty = False  # approximation

    @property
    def seat_wind(self):
        return self.my_pid

    def init_from_requests(self, requests):
        """Reconstruct state by replaying all requests/responses."""
        if not requests:
            return

        # Request 0: init  "0 playerID prevalentWind"
        parts = requests[0].split()
        self.my_pid = int(parts[1])
        self.prevalent_wind = int(parts[2])

    def apply_deal(self, deal_str):
        """Request 1: deal  "1 f0 f1 f2 f3 t1..t13 [flowers...]" """
        parts = deal_str.split()
        flowers_per_player = [int(parts[i]) for i in range(1, 5)]
        self.hand = list(parts[5:18])
        self.flower_count = flowers_per_player[self.my_pid]

    def apply_draw(self, tile):
        """I drew a tile."""
        self.last_draw = tile
        self.hand.append(tile)
        self.last_bugang = False

    def apply_my_play(self, tile):
        if tile in self.hand:
            self.hand.remove(tile)

    def apply_my_gang(self, tile):
        """Concealed kong (暗杠)."""
        for _ in range(4):
            if tile in self.hand:
                self.hand.remove(tile)
        self.packs.append(("GANG", tile, self.my_pid))

    def apply_my_bugang(self, tile):
        """Supplement kong (补杠)."""
        if tile in self.hand:
            self.hand.remove(tile)
        for i, p in enumerate(self.packs):
            if p[0] == "PENG" and p[1] == tile:
                self.packs[i] = ("GANG", tile, p[2])
                break

    def apply_notify(self, notify_str):
        """
        Stage-3 notification: "3 pid ACTION [tile1] [tile2]"
        Returns (action, pid, tile1, tile2).
        Also tracks state changes for my player's reactions.
        """
        parts = notify_str.split()
        pid = int(parts[1])
        action = parts[2]
        tile1 = parts[3] if len(parts) > 3 else None
        tile2 = parts[4] if len(parts) > 4 else None

        if action == "PLAY":
            self.last_discard_pid = pid
            self.last_discard_tile = tile1
            self.last_bugang = False
        elif action == "PENG":
            if pid == self.my_pid:
                # I penged — remove 2 copies from hand
                tile = tile1
                for _ in range(2):
                    if tile in self.hand:
                        self.hand.remove(tile)
                self.packs.append(("PENG", tile, self.last_discard_pid))
        elif action == "CHI":
            if pid == self.my_pid:
                mid_tile = tile1  # middle tile of the chi sequence
                discard_after = tile2
                # Remove the 3 chi tiles from hand (discard_tile comes from others)
                # chi tiles: mid-1, mid, mid+1; one is from last discard
                if mid_tile and mid_tile[0] in ('W', 'B', 'T'):
                    n = int(mid_tile[1])
                    for delta in (-1, 0, 1):
                        t = f"{mid_tile[0]}{n + delta}"
                        if t != self.last_discard_tile and t in self.hand:
                            self.hand.remove(t)
                    # Add the discarded tile temporarily, then remove the sequence
                    # The peng adds "last_discard_tile" to hand first
                self.packs.append(("CHI", mid_tile, 1))  # offer=1 means left player
        elif action == "GANG":
            if pid == self.my_pid:
                # I declared gang from discard (opponent had 3 of them)
                pass
            self.last_bugang = False
        elif action == "BUGANG":
            self.last_discard_tile = tile1
            self.last_discard_pid = pid
            self.last_bugang = (pid != self.my_pid)

        return action, pid, tile1, tile2

    def apply_my_peng(self, discard_tile, my_discard):
        """I responded to a discard with PENG + my_discard."""
        for _ in range(2):
            if discard_tile in self.hand:
                self.hand.remove(discard_tile)
        self.packs.append(("PENG", discard_tile, self.last_discard_pid if self.last_discard_pid is not None else 0))
        if my_discard in self.hand:
            self.hand.remove(my_discard)

    def apply_my_chi(self, mid_tile, my_discard, discard_tile):
        """I responded with CHI mid_tile my_discard."""
        n = int(mid_tile[1])
        suit = mid_tile[0]
        for delta in (-1, 0, 1):
            t = f"{suit}{n + delta}"
            if t == discard_tile:
                continue
            if t in self.hand:
                self.hand.remove(t)
        offer = discard_tile[1] if discard_tile else '1'
        self.packs.append(("CHI", mid_tile, int(mid_tile[1]) - int(discard_tile[1]) + 1))
        if my_discard in self.hand:
            self.hand.remove(my_discard)

    def apply_my_meld_gang(self, discard_tile):
        """I responded to a discard with GANG (have 3 in hand)."""
        for _ in range(3):
            if discard_tile in self.hand:
                self.hand.remove(discard_tile)
        self.packs.append(("GANG", discard_tile, self.last_discard_pid if self.last_discard_pid is not None else 0))

    def can_chi(self, discard_tile, next_pid):
        """Can I (my_pid == next_pid) chi the discard_tile?"""
        if self.my_pid != next_pid:
            return []
        if not is_number_tile(discard_tile):
            return []
        suit = discard_tile[0]
        n = int(discard_tile[1])
        options = []
        # discard_tile is one of {mid-1, mid, mid+1} in the sequence
        for mid_offset in (-1, 0, 1):
            mid_n = n + mid_offset
            if mid_n < 2 or mid_n > 8:
                continue
            needed = []
            for delta in (-1, 0, 1):
                t = f"{suit}{mid_n + delta}"
                if t != discard_tile:
                    needed.append(t)
            if all(t in self.hand for t in needed):
                options.append(f"{suit}{mid_n}")
        return options

    def can_peng(self, discard_tile):
        """Can I peng the discard_tile?"""
        return self.hand.count(discard_tile) >= 2

    def can_meld_gang(self, discard_tile):
        """Can I declare GANG from a discard (have 3 copies)?"""
        return self.hand.count(discard_tile) >= 3

    def can_angang(self):
        """Tiles I can declare concealed kong (暗杠) with."""
        result = []
        for t in set(self.hand):
            if self.hand.count(t) == 4:
                result.append(t)
        return result

    def can_bugang(self):
        """Tiles I can supplement kong (补杠) with."""
        result = []
        penged = {p[1] for p in self.packs if p[0] == "PENG"}
        for t in penged:
            if t in self.hand:
                result.append(t)
        return result


# ── Decision engine ───────────────────────────────────────────────────────────

def decide_after_draw(state):
    """
    Called when I just drew a tile (state.last_draw is set).
    Returns response string.
    """
    hand = state.hand  # includes the drawn tile
    packs = state.packs
    drawn = state.last_draw

    # 1. Check if we can HU (自摸)
    hand_without_draw = [t for t in hand if t != drawn]
    # Actually, for HU check, drawn tile is win_tile
    # hand_tiles = everything except win_tile
    hand_except_win = list(hand)
    hand_except_win.remove(drawn)
    fan = check_hu(
        hand_except_win, packs, drawn,
        seat_wind=state.seat_wind,
        prevalent_wind=state.prevalent_wind,
        is_self_drawn=True,
        flower_count=state.flower_count
    )
    if fan >= 8:
        return "HU"

    # 2. Check BUGANG (补杠): upgrade PENG to GANG
    bugang_tiles = state.can_bugang()
    for bt in bugang_tiles:
        # Check if BUGANG doesn't break tenpai
        test_hand = list(hand)
        test_hand.remove(bt)
        s_before, _ = shanten(list(hand), packs)
        s_after, _ = shanten(test_hand, packs)
        if s_after <= s_before:
            return f"BUGANG {bt}"

    # 3. Check ANGANG (暗杠): concealed kong
    angang_tiles = state.can_angang()
    for at in angang_tiles:
        # Check if ANGANG doesn't break tenpai
        test_hand = list(hand)
        for _ in range(4):
            test_hand.remove(at)
        s_before, _ = shanten(list(hand), packs)
        s_after, _ = shanten(test_hand, packs)
        if s_after <= s_before:
            return f"GANG {at}"

    # 4. Discard the tile that minimizes shanten
    tile_to_discard = best_discard(list(hand), packs)
    return f"PLAY {tile_to_discard}"


def decide_after_discard(state, discard_pid):
    """
    Called when another player (discard_pid) discarded a tile.
    Returns response string.
    """
    discard_tile = state.last_discard_tile
    hand = state.hand
    packs = state.packs
    next_pid = (discard_pid + 1) % 4

    # 1. Check if we can HU (荣和)
    fan = check_hu(
        list(hand), packs, discard_tile,
        seat_wind=state.seat_wind,
        prevalent_wind=state.prevalent_wind,
        is_self_drawn=False,
        flower_count=state.flower_count
    )
    if fan >= 8:
        return "HU"

    # 2. Check GANG (from discard, have 3 in hand)
    if state.can_meld_gang(discard_tile):
        # GANG if it doesn't hurt too much
        test_hand = list(hand)
        for _ in range(3):
            test_hand.remove(discard_tile)
        s_before, _ = shanten(list(hand), packs)
        s_after, _ = shanten(test_hand, packs + [("GANG", discard_tile, discard_pid)])
        if s_after <= s_before:
            return "GANG"

    # 3. Check PENG (have 2 in hand)
    if state.can_peng(discard_tile):
        test_hand = list(hand)
        test_hand.remove(discard_tile)
        test_hand.remove(discard_tile)
        test_packs = packs + [("PENG", discard_tile, discard_pid)]
        s_before, _ = shanten(list(hand), packs)
        # After peng, what's the best tile to discard?
        discard_after = best_discard(test_hand, test_packs)
        if discard_after:
            test_hand_after = [t for t in test_hand if t != discard_after]
            test_hand_after_once = list(test_hand)
            test_hand_after_once.remove(discard_after)
            s_after, _ = shanten(test_hand_after_once, test_packs)
            if s_after < s_before:
                return f"PENG {discard_after}"

    # 4. Check CHI (only if I'm the next player)
    chi_options = state.can_chi(discard_tile, next_pid)
    if chi_options:
        for mid_tile in chi_options:
            n = int(mid_tile[1])
            suit = mid_tile[0]
            # Build test hand after chi
            test_hand = list(hand)
            for delta in (-1, 0, 1):
                t = f"{suit}{n + delta}"
                if t != discard_tile and t in test_hand:
                    test_hand.remove(t)
            test_packs = packs + [("CHI", mid_tile, 1)]
            s_before, _ = shanten(list(hand), packs)
            discard_after = best_discard(test_hand, test_packs)
            if discard_after:
                test_hand_after = list(test_hand)
                test_hand_after.remove(discard_after)
                s_after, _ = shanten(test_hand_after, test_packs)
                if s_after < s_before:
                    return f"CHI {mid_tile} {discard_after}"

    return "PASS"


def decide_after_gang_notify(state):
    """
    Called after BUGANG notification (can potentially rob the kong / 抢杠和).
    """
    if not state.last_bugang:
        return "PASS"
    discard_tile = state.last_discard_tile
    if discard_tile is None:
        return "PASS"
    fan = check_hu(
        list(state.hand), state.packs, discard_tile,
        seat_wind=state.seat_wind,
        prevalent_wind=state.prevalent_wind,
        is_self_drawn=False,
        is_about_kong=True,
        flower_count=state.flower_count
    )
    if fan >= 8:
        return "HU"
    return "PASS"


# ── Main protocol handler ─────────────────────────────────────────────────────

def run():
    raw = sys.stdin.read()
    data = json.loads(raw)

    requests = []
    responses = []
    turn_id = len(data.get("responses", []))
    for i in range(turn_id):
        requests.append(data["requests"][i])
        responses.append(data["responses"][i])
    requests.append(data["requests"][turn_id])

    state = GameState()

    if turn_id == 0:
        # First request: just init, output PASS
        state.init_from_requests(requests)
        output_response("PASS")
        return

    # Reconstruct state by replaying history
    state.init_from_requests(requests)

    # Request 1: deal
    state.apply_deal(requests[1])

    # Process turns 2..turn_id-1 (history)
    i = 2
    while i < turn_id:
        req = requests[i]
        resp = responses[i]
        parts = req.split()
        rtype = int(parts[0])

        if rtype == 2:
            # My draw
            tile = parts[1]
            state.apply_draw(tile)
            resp_parts = resp.split()
            if resp_parts[0] == "PLAY":
                state.apply_my_play(resp_parts[1])
            elif resp_parts[0] == "GANG":
                state.apply_my_gang(resp_parts[1])
            elif resp_parts[0] == "BUGANG":
                state.apply_my_bugang(resp_parts[1])
            elif resp_parts[0] == "HU":
                pass
            elif resp_parts[0] == "PASS":
                pass
        elif rtype == 3:
            action, pid, tile1, tile2 = state.apply_notify(req)
            # Check if I responded to this event
            resp_parts = resp.split()
            if resp_parts[0] == "PENG" and len(resp_parts) >= 2:
                state.apply_my_peng(tile1, resp_parts[1])
            elif resp_parts[0] == "CHI" and len(resp_parts) >= 3:
                state.apply_my_chi(resp_parts[1], resp_parts[2], tile1)
            elif resp_parts[0] == "GANG" and action == "PLAY":
                state.apply_my_meld_gang(tile1)
        i += 1

    # Now handle the current request
    current_req = requests[turn_id]
    parts = current_req.split()
    rtype = int(parts[0])

    if rtype == 2:
        # My draw
        tile = parts[1]
        state.apply_draw(tile)
        response = decide_after_draw(state)
    elif rtype == 3:
        action, pid, tile1, tile2 = state.apply_notify(current_req)
        if pid == state.my_pid:
            # I was the one who played — must PASS
            response = "PASS"
        elif action == "PLAY":
            response = decide_after_discard(state, pid)
        elif action in ("GANG", "BUGANG"):
            response = decide_after_gang_notify(state)
        else:
            response = "PASS"
    else:
        response = "PASS"

    output_response(response)


def output_response(resp):
    out = {"response": resp}
    print(json.dumps(out))


if __name__ == "__main__":
    run()
