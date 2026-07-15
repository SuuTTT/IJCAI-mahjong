"""Chinese Standard Mahjong (MCR / 国标麻将) engine for the Ludus platform.

Competition wall variant: the shuffled 136-tile wall is split into four
private 34-tile segments, one per seat; each player draws exclusively from
their own segment. The MahjongGB C-extension (PyMahjongGB) is the ground-truth
fan calculator and win validator.

Agents follow the Botzone protocol: an object exposing ``respond(request:str) -> str``.
"""

import random

try:
    from MahjongGB import MahjongFanCalculator
except Exception:  # pragma: no cover - only importable on the test box
    MahjongFanCalculator = None


# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------
SUITS = "WTB"  # characters(万) / bamboo(条) / dots(筒)


def all_tile_codes():
    """The 34 distinct tile codes."""
    codes = []
    for s in SUITS:
        for n in range(1, 10):
            codes.append(f"{s}{n}")
    for n in range(1, 5):
        codes.append(f"F{n}")  # winds 东南西北
    for n in range(1, 4):
        codes.append(f"J{n}")  # dragons 中发白
    return codes


def build_wall(seed=0):
    """Deterministic shuffled 136-tile wall from ``seed``."""
    rng = random.Random(seed)
    wall = []
    for code in all_tile_codes():
        wall.extend([code] * 4)
    rng.shuffle(wall)
    return wall


def _is_suited(tile):
    return tile[0] in SUITS


def _rank(tile):
    return int(tile[1])


def meld_tiles(meld):
    """Physical tiles composing an exposed meld (for conservation counting)."""
    kind, tile, _offer = meld
    if kind == "CHI":
        s, n = tile[0], int(tile[1])
        return [f"{s}{n-1}", f"{s}{n}", f"{s}{n+1}"]
    if kind == "PENG":
        return [tile] * 3
    if kind == "GANG":
        return [tile] * 4
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class _Game:
    def __init__(self, agents, wall, quan, seed):
        assert len(agents) == 4
        assert len(wall) == 136
        self.agents = agents
        self.quan = quan
        self.rng = random.Random(seed * 7919 + 13)
        self.segments = [wall[0:34], wall[34:68], wall[68:102], wall[102:136]]
        self.ptr = [13, 13, 13, 13]
        self.hands = [list(self.segments[s][:13]) for s in range(4)]
        self.melds = [[] for _ in range(4)]   # list of (kind, tile, offer)
        self.discards = [[] for _ in range(4)]
        self.log = []
        self.frames = []      # per-step board snapshots for the replay viewer

    # -- low level helpers -------------------------------------------------
    def _snap(self, turn, mode):
        """Full-board snapshot (replay shows all hands). One per driver step."""
        last = next((e for e in reversed(self.log) if isinstance(e, str)), "")
        return {
            "hands": [sorted(h) for h in self.hands],
            "melds": [[list(m) for m in ms] for ms in self.melds],
            "discards": [list(d) for d in self.discards],
            "turn": turn, "mode": mode, "last": last,
            "wall_left": [self._wall_left(s) for s in range(4)],
        }

    def _send(self, seat, request):
        return (self.agents[seat].respond(request) or "").strip()

    def _broadcast_others(self, actor, request):
        """Send request to the three players that are not ``actor``; ignore replies."""
        for i in (1, 2, 3):
            self._send((actor + i) % 4, request)

    def _draw(self, seat):
        """Draw the next tile from ``seat``'s own segment, or None if exhausted."""
        if self.ptr[seat] >= 34:
            return None
        tile = self.segments[seat][self.ptr[seat]]
        self.ptr[seat] += 1
        return tile

    def _wall_left(self, seat):
        return 34 - self.ptr[seat]

    def _pick_legal_discard(self, seat):
        return self.rng.choice(self.hands[seat])

    def _relative_offer(self, claimant, discarder):
        # relative seat the claimed tile came from: 1..3, never 0
        return (discarder - claimant) % 4

    # -- win validation ----------------------------------------------------
    def _try_win(self, seat, win_tile, is_self, about_kong):
        """Return total fan (>=8) if ``seat`` legitimately wins, else None."""
        concealed = list(self.hands[seat])
        if is_self:
            if win_tile not in concealed:
                return None
            concealed.remove(win_tile)
        pack = tuple((m[0], m[1], m[2]) for m in self.melds[seat])
        try:
            res = MahjongFanCalculator(
                pack=pack,
                hand=tuple(concealed),
                winTile=win_tile,
                flowerCount=0,
                isSelfDrawn=is_self,
                is4thTile=False,
                isAboutKong=about_kong,
                isWallLast=False,
                seatWind=seat,
                prevalentWind=self.quan,
            )
        except Exception:
            return None
        fan = sum(v for v, _n in res)
        if fan < 8:
            return None
        return fan

    # -- scoring -----------------------------------------------------------
    @staticmethod
    def _score_zimo(winner, fan):
        scores = {}
        for s in range(4):
            scores[s] = 3 * (8 + fan) if s == winner else -(8 + fan)
        return scores

    @staticmethod
    def _score_rong(winner, discarder, fan):
        scores = {}
        for s in range(4):
            if s == winner:
                scores[s] = 24 + fan
            elif s == discarder:
                scores[s] = -(8 + fan)
            else:
                scores[s] = -8
        return scores

    def _result(self, ending, winner=None, fan=None, discarder=None):
        if ending == "draw":
            scores = {s: 0 for s in range(4)}
        elif ending == "zimo":
            scores = self._score_zimo(winner, fan)
        elif ending == "rong":
            scores = self._score_rong(winner, discarder, fan)
        else:
            raise ValueError(ending)
        self.log.append({"event": "end", "ending": ending, "winner": winner,
                         "fan": fan, "scores": scores})
        self.frames.append(self._snap(winner if winner is not None else -1, ending))
        self.frames[-1]["scores"] = scores          # terminal frame carries the result
        final = {
            "hands": [list(h) for h in self.hands],
            "melds": [[list(m) for m in ms] for ms in self.melds],
            "discards": [list(d) for d in self.discards],
            "wall_remaining": [self.segments[s][self.ptr[s]:] for s in range(4)],
        }
        return {"scores": scores, "winner": winner, "fan": fan, "ending": ending,
                "log": self.log, "frames": self.frames, "_final": final}

    # -- setup -------------------------------------------------------------
    def _init_agents(self):
        for s in range(4):
            self._send(s, f"0 {s} {self.quan}")
        for s in range(4):
            self._send(s, "1 0 0 0 0 " + " ".join(self.hands[s]))
        self.log.append({"event": "init", "quan": self.quan})
        self.log.append({"event": "deal", "hands": [list(h) for h in self.hands]})

    # -- main driver -------------------------------------------------------
    def run(self):
        self._init_agents()
        turn = 0            # seat that must draw & act
        kong = False        # whether this draw is a kong replacement
        mode = "draw"
        pending = None      # (discarder, tile, event_str) for resolve mode

        while True:
            self.frames.append(self._snap(turn, mode))
            if mode == "draw":
                tile = self._draw(turn)
                if tile is None:
                    return self._result("draw")
                self.hands[turn].append(tile)
                self.log.append(f"DRAW {turn} {tile}")
                resp = self._send(turn, f"2 {tile}")
                self._broadcast_others(turn, f"3 {turn} DRAW")
                out = self._self_action(turn, tile, resp, kong)
                kind = out[0]
                if kind == "win":
                    return self._result("zimo", winner=turn, fan=out[1])
                if kind == "rob_win":  # bugang robbed
                    return self._result("rong", winner=out[1], fan=out[2],
                                        discarder=turn)
                if kind == "kong":     # concealed gang or added kong -> replacement
                    kong = True
                    mode = "draw"      # same turn draws replacement
                    continue
                # discard
                _, dtile = out
                self.discards[turn].append(dtile)
                pending = (turn, dtile, f"3 {turn} PLAY {dtile}")
                mode = "resolve"
                kong = False
                continue

            else:  # resolve a claimable discard
                discarder, dtile, event = pending
                out = self._resolve(discarder, dtile, event)
                kind = out[0]
                if kind == "win":
                    return self._result("rong", winner=out[1], fan=out[2],
                                        discarder=discarder)
                if kind == "pass":
                    turn = (discarder + 1) % 4
                    kong = False
                    mode = "draw"
                    continue
                if kind == "gang":     # exposed kong claimant draws replacement
                    turn = out[1]
                    kong = True
                    mode = "draw"
                    continue
                if kind == "chain":    # peng/chi claimant re-discarded
                    pending = out[1]
                    mode = "resolve"
                    continue

    # -- self action on own draw ------------------------------------------
    def _self_action(self, seat, drawn, resp, kong):
        parts = resp.split()
        act = parts[0].upper() if parts else "PLAY"

        if act == "HU":
            fan = self._try_win(seat, drawn, is_self=True, about_kong=kong)
            if fan is not None:
                self.log.append(f"HU {seat} {drawn} zimo fan={fan}")
                return ("win", fan)
            self.log.append(f"ILLEGAL_HU {seat} {drawn}")
            return ("discard", self._forced_discard(seat, drawn))

        if act == "GANG" and len(parts) >= 2:      # concealed kong
            t = parts[1]
            if self.hands[seat].count(t) == 4:
                for _ in range(4):
                    self.hands[seat].remove(t)
                self.melds[seat].append(("GANG", t, 0))
                self.log.append(f"3 {seat} GANG")
                self._broadcast_others(seat, f"3 {seat} GANG")
                return ("kong",)
            self.log.append(f"ILLEGAL_GANG {seat} {t}")
            return ("discard", self._forced_discard(seat, drawn))

        if act == "BUGANG" and len(parts) >= 2:     # added kong
            t = parts[1]
            peng_idx = next((i for i, m in enumerate(self.melds[seat])
                             if m[0] == "PENG" and m[1] == t), None)
            if peng_idx is not None and t in self.hands[seat]:
                self.hands[seat].remove(t)
                offer = self.melds[seat][peng_idx][2] or 1
                self.melds[seat][peng_idx] = ("GANG", t, offer)
                self.log.append(f"3 {seat} BUGANG {t}")
                # offer rob-the-kong to the other three, in seat order after seat
                robber = None
                fan = None
                for i in (1, 2, 3):
                    p = (seat + i) % 4
                    r = self._send(p, f"3 {seat} BUGANG {t}").split()
                    if robber is None and r and r[0].upper() == "HU":
                        f = self._try_win(p, t, is_self=False, about_kong=True)
                        if f is not None:
                            robber, fan = p, f
                if robber is not None:
                    self.log.append(f"HU {robber} {t} robkong fan={fan}")
                    return ("rob_win", robber, fan)
                return ("kong",)
            self.log.append(f"ILLEGAL_BUGANG {seat} {t}")
            return ("discard", self._forced_discard(seat, drawn))

        # PLAY <tile> (or anything unrecognised)
        if act == "PLAY" and len(parts) >= 2 and parts[1] in self.hands[seat]:
            t = parts[1]
            self.hands[seat].remove(t)
            self.log.append(f"3 {seat} PLAY {t}")
            return ("discard", t)
        # illegal / missing discard tile
        t = self._forced_discard(seat, drawn)
        return ("discard", t)

    def _forced_discard(self, seat, drawn):
        if drawn not in self.hands[seat]:
            # shouldn't happen, but stay safe
            t = self._pick_legal_discard(seat)
        else:
            t = self._pick_legal_discard(seat)
        self.hands[seat].remove(t)
        self.log.append(f"3 {seat} PLAY {t}")
        return t

    # -- resolve claims on a discard --------------------------------------
    def _resolve(self, discarder, tile, event):
        self.log.append(event)
        responders = [(discarder + i) % 4 for i in (1, 2, 3)]
        claims = {}
        for p in responders:
            claims[p] = self._send(p, event).split()

        # Priority 1: HU (rong), in seat order after discarder
        for p in responders:
            c = claims[p]
            if c and c[0].upper() == "HU":
                fan = self._try_win(p, tile, is_self=False, about_kong=False)
                if fan is not None:
                    # discard consumed by the win
                    if self.discards[discarder] and self.discards[discarder][-1] == tile:
                        self.discards[discarder].pop()
                    self.log.append(f"HU {p} {tile} rong fan={fan}")
                    return ("win", p, fan)

        # Priority 2: PENG / exposed GANG (any other player)
        for p in responders:
            c = claims[p]
            if not c:
                continue
            a = c[0].upper()
            if a == "GANG" and self.hands[p].count(tile) >= 3 and self._wall_left(p) > 0:
                self._consume_discard(discarder, tile)
                for _ in range(3):
                    self.hands[p].remove(tile)
                self.melds[p].append(("GANG", tile, self._relative_offer(p, discarder)))
                self.log.append(f"3 {p} GANG")
                self._notify_claim(p, f"3 {p} GANG")
                return ("gang", p)
            if a == "PENG" and self.hands[p].count(tile) >= 2:
                self._consume_discard(discarder, tile)
                for _ in range(2):
                    self.hands[p].remove(tile)
                self.melds[p].append(("PENG", tile, self._relative_offer(p, discarder)))
                dtile = self._claim_discard(p, c[1] if len(c) >= 2 else None)
                self.discards[p].append(dtile)
                ev = f"3 {p} PENG {dtile}"
                self.log.append(ev)
                return ("chain", (p, dtile, ev))

        # Priority 3: CHI (only the next player, suited tiles)
        nxt = (discarder + 1) % 4
        c = claims[nxt]
        if c and c[0].upper() == "CHI" and len(c) >= 3 and _is_suited(tile):
            mid = c[1]
            if self._can_chi(nxt, tile, mid):
                self._consume_discard(discarder, tile)
                s = mid[0]
                n = int(mid[1])
                needed = [f"{s}{n-1}", f"{s}{n}", f"{s}{n+1}"]
                needed.remove(tile)  # remove the offered tile; keep the two from hand
                for t2 in needed:
                    self.hands[nxt].remove(t2)
                offer = _rank(tile) - n + 2  # 1/2/3 position of claimed tile
                self.melds[nxt].append(("CHI", mid, offer))
                dtile = self._claim_discard(nxt, c[2])
                self.discards[nxt].append(dtile)
                ev = f"3 {nxt} CHI {mid} {dtile}"
                self.log.append(ev)
                return ("chain", (nxt, dtile, ev))

        return ("pass",)

    def _consume_discard(self, discarder, tile):
        if self.discards[discarder] and self.discards[discarder][-1] == tile:
            self.discards[discarder].pop()

    def _notify_claim(self, actor, event):
        for i in (1, 2, 3):
            self._send((actor + i) % 4, event)

    def _claim_discard(self, seat, wanted):
        """After a peng/chi, seat discards ``wanted`` if legal else a random tile."""
        if wanted is not None and wanted in self.hands[seat]:
            self.hands[seat].remove(wanted)
            return wanted
        self.log.append(f"ILLEGAL_CLAIM_DISCARD {seat} {wanted}")
        t = self._pick_legal_discard(seat)
        self.hands[seat].remove(t)
        return t

    def _can_chi(self, seat, tile, mid):
        if not (_is_suited(mid) and mid[0] == tile[0]):
            return False
        s = mid[0]
        n = int(mid[1])
        if n < 2 or n > 8:
            return False
        run = [f"{s}{n-1}", f"{s}{n}", f"{s}{n+1}"]
        if tile not in run:
            return False
        others = [t for t in run if t != tile]
        hand = list(self.hands[seat])
        for t2 in others:
            if t2 in hand:
                hand.remove(t2)
            else:
                return False
        return True


def calc_fan_raw(pack, hand, win_tile, is_self, about_kong, is_wall_last,
                 is_4th, seat, quan, flower=0, verbose=False):
    """Raw fan list from PyMahjongGB.

    Non-verbose: tuple of ``(value, name)``.  Verbose: tuple of
    ``(point, cnt, name, name_en)`` — this is what the Botzone judge uses, so
    identical fans stay split (value = per-unit point, cnt = multiplicity)
    instead of being combined.  Shared by :class:`_Game._try_win` and the
    validator adapter.  Raises on a non-winning hand.
    """
    return MahjongFanCalculator(
        pack=tuple(pack), hand=tuple(hand), winTile=win_tile,
        flowerCount=flower, isSelfDrawn=is_self, is4thTile=is_4th,
        isAboutKong=about_kong, isWallLast=is_wall_last,
        seatWind=seat, prevalentWind=quan, verbose=verbose)


def play_game(agents, wall=None, quan=0, seed=0):
    """Play one MCR game and return the result dict.

    Returns ``{"scores": {0..3:int}, "winner": seat|None, "fan": int|None,
    "ending": "zimo"|"rong"|"draw", "log": [...]}``.
    """
    if wall is None:
        wall = build_wall(seed)
    else:
        wall = list(wall)
        assert len(wall) == 136, "wall must be 136 tiles"
    game = _Game(agents, wall, quan, seed)
    return game.run()
