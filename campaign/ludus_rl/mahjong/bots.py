"""Botzone-protocol agents for the MCR engine."""

import random


class RandomLegalBot:
    """A minimal Botzone-protocol agent.

    Tracks its own concealed hand from the request stream. On its own draw it
    discards a uniformly-random legal tile. It never claims (always PASS on
    other players' discards) and never declares HU/GANG/BUGANG. This keeps the
    hand a valid legal hand at all times, so games run to exhaustion (or, in the
    rare degenerate case, someone else could act — but this bot never wins).
    """

    def __init__(self, seed=None):
        self.seat = None
        self.quan = None
        self.hand = []
        self.rng = random.Random(seed)

    def respond(self, request):
        parts = request.split()
        if not parts:
            return "PASS"
        code = parts[0]

        if code == "0":
            # 0 <seat> <prevalentWind>
            self.seat = int(parts[1])
            self.quan = int(parts[2])
            return "PASS"

        if code == "1":
            # 1 <f0..f3> <t0..t12>
            self.hand = list(parts[5:5 + 13])
            return "PASS"

        if code == "2":
            # own draw: 2 <tile>
            tile = parts[1]
            self.hand.append(tile)
            discard = self.rng.choice(self.hand)
            self.hand.remove(discard)
            return f"PLAY {discard}"

        # code == "3": some player did something; this bot never claims.
        return "PASS"


class EfficiencyBot:
    """Stronger baseline that still never mutates its hand via claims (so it is
    provably consistent with the engine): it keeps pairs/runs, discards the most
    isolated tile, and DECLARES any valid >=8-fan win — self-draw or off an
    opponent's discard — by asking the real fan calculator. Declaring HU ends the
    game, so no peng/chi/gang bookkeeping is needed. Won't reliably reach the
    8-fan bar (that needs fan-aware play, i.e. the kdens3 champion), but plays
    tidily and takes wins when they appear."""

    def __init__(self, seed=None):
        self.seat = 0
        self.quan = 0
        self.hand = []
        self.rng = random.Random(seed)

    def _fan(self, concealed, wintile, is_self):
        try:
            from MahjongGB import MahjongFanCalculator
            r = MahjongFanCalculator(
                pack=(), hand=tuple(concealed), winTile=wintile, flowerCount=0,
                isSelfDrawn=is_self, is4thTile=False, isAboutKong=False,
                isWallLast=False, seatWind=self.seat, prevalentWind=self.quan)
            return sum(v for v, _ in r)
        except Exception:
            return None      # ERROR_NOT_WIN (or calculator unavailable)

    def _keep_value(self, t, counts):
        v = 2 * (counts[t] - 1)                      # duplicates -> pairs/triplets
        if t[0] in "WTB":
            s, r = t[0], int(t[1])
            for dr in (-2, -1, 1, 2):
                if 1 <= r + dr <= 9 and counts.get(f"{s}{r + dr}", 0) > 0:
                    v += 2 - abs(dr)                 # nearer neighbours worth more
        return v

    def _eff_discard(self):
        from collections import Counter
        counts = Counter(self.hand)
        # lowest keep-value; break ties by dumping honours/terminals first
        return min(self.hand, key=lambda t: (
            self._keep_value(t, counts),
            0 if t[0] in "FJ" else (0 if t[1] in "19" else 1)))

    def respond(self, request):
        parts = request.split()
        if not parts:
            return "PASS"
        code = parts[0]
        if code == "0":
            self.seat = int(parts[1]); self.quan = int(parts[2]); return "PASS"
        if code == "1":
            self.hand = list(parts[5:5 + 13]); return "PASS"
        if code == "2":
            tile = parts[1]; self.hand.append(tile)
            concealed = list(self.hand); concealed.remove(tile)
            fan = self._fan(concealed, tile, True)
            if fan is not None and fan >= 8:
                return "HU"
            d = self._eff_discard(); self.hand.remove(d)
            return f"PLAY {d}"
        if (code == "3" and len(parts) >= 4 and parts[2] == "PLAY"
                and int(parts[1]) != self.seat):
            t = parts[3]
            fan = self._fan(list(self.hand), t, False)
            if fan is not None and fan >= 8:
                return "HU"
            return "PASS"
        return "PASS"
