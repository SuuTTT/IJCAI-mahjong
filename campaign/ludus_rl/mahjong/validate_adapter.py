"""Adapter exposing the MCR engine to the IJCAI-2026 correctness validator.

The validator (``mcr_testset/validate_engine.py``) drives an engine through a
recorded game turn by turn: it calls :meth:`MyEngine.reset` to get the first
(INIT) turn, then repeatedly calls :meth:`MyEngine.step` feeding the four
seats' recorded response strings, and requires the engine to reproduce the
judge's protocol stream exactly — every per-seat request string, every display
event (draws, claim resolutions, gang discrimination, terminal fan/score).

The engine does NOT decide moves; the recorded responses are fed in and the
engine resolves them into the exact next judge event.  Rule logic (tile order,
claim priority, PyMahjongGB fan calculation, scoring) is shared with
``mahjong/engine.py`` via :func:`mahjong.engine.calc_fan_raw`.

Importable as ``mahjong.validate_adapter:MyEngine``.
"""

from mahjong.engine import calc_fan_raw, meld_tiles

# Claim arbitration priority: HU beats PENG/GANG beats CHI.
_PRIO = {"HU": 3, "GANG": 2, "PENG": 2, "CHI": 1}


class MyEngine:
    # ------------------------------------------------------------------ API
    def reset(self, wall, quan, srand):
        self._gen = self._play(list(wall), int(quan), int(srand))
        return next(self._gen)

    def step(self, responses):
        return self._gen.send(dict(responses or {}))

    # -------------------------------------------------------------- helpers
    def _draw_display(self, s, t, kong):
        return {"action": "DRAW", "player": s, "tile": t,
                "canHu": self._canhu_draw(s, t, kong),
                "tileCnt": list(self.remain)}

    # ------------------------------------------------------------------ canHu
    # The judge attaches a per-seat win-eligibility array ``canHu[4]`` to every
    # display event: a value >= 0 is the exact fan total that seat could win
    # with on the current tile event (even below the 8-fan legal minimum);
    # -3 means the seat cannot form a winning hand on it; -4 means not
    # applicable (own event / no tile in flight).  INIT is all 0; DEAL, GANG,
    # HU and HUANG are all -4.  The winning event's canHu[winner] equals the
    # terminal fanCnt, so this reuses exactly the fan path of :meth:`_hu_turn`.

    def _fan_total(self, seat, win_tile, is_self, about_kong, wall_last):
        """Fan total ``seat`` would score winning on ``win_tile`` now, or -3."""
        hand = list(self.hands[seat])
        if is_self:
            hand.remove(win_tile)
        is_4th = self._is_4th_tile(win_tile, seat)
        try:
            raw = calc_fan_raw(self.packs[seat], hand, win_tile, is_self,
                               about_kong, wall_last, is_4th, seat, self.quan,
                               verbose=True)
        except Exception:
            return -3
        return sum(point * cnt for point, cnt, *_ in raw)

    def _canhu_draw(self, turn, t, kong):
        """canHu for a DRAW display: only the drawer is checked (zimo)."""
        wall_last = self.remain[(turn + 1) % 4] == 0
        ch = [-4] * 4
        ch[turn] = self._fan_total(turn, t, True, kong, wall_last)
        return ch

    def _canhu_discard(self, disc_seat, out):
        """canHu for a PLAY/PENG/CHI display: the three other seats are
        checked for a ron on the (embedded) discard ``out``."""
        wall_last = self.remain[(disc_seat + 1) % 4] == 0
        self.discards[disc_seat].pop()      # match the ron-time table state
        ch = [-4] * 4
        for ss in range(4):
            if ss != disc_seat:
                ch[ss] = self._fan_total(ss, out, False, False, wall_last)
        self.discards[disc_seat].append(out)
        return ch

    def _canhu_bugang(self, turn, bt):
        """canHu for a BUGANG display: qianggang (robbing the kong) check."""
        wall_last = self.remain[(turn + 1) % 4] == 0
        ch = [-4] * 4
        for ss in range(4):
            if ss != turn:
                ch[ss] = self._fan_total(ss, bt, False, True, wall_last)
        return ch

    def _turn(self, requests, display):
        return {"requests": requests, "display": display}

    def _all_req(self, s):
        return None  # placeholder, never used

    # ------------------------------------------------------------- fan/score
    def _fan_display(self, winner, win_tile, hand, packs, is_self, about_kong,
                     wall_last, is_4th):
        """Compute the terminal HU display's fan list, fanCnt and score."""
        raw = calc_fan_raw(packs, hand, win_tile, is_self, about_kong,
                           wall_last, is_4th, winner, self.quan, verbose=True)
        # verbose entries: (point, cnt, name, name_en) — matches the judge's
        # per-unit point/multiplicity split exactly.
        fan = [{"cnt": cnt, "name": name, "value": point}
               for point, cnt, name, *_ in raw]
        fan_cnt = sum(point * cnt for point, cnt, name, *_ in raw)
        return fan, fan_cnt

    def _score_zimo(self, winner, fan_cnt):
        return [3 * (8 + fan_cnt) if s == winner else -(8 + fan_cnt)
                for s in range(4)]

    def _score_ron(self, winner, discarder, fan_cnt):
        sc = []
        for s in range(4):
            if s == winner:
                sc.append(3 * 8 + fan_cnt)
            elif s == discarder:
                sc.append(-(8 + fan_cnt))
            else:
                sc.append(-8)
        return sc

    def _hu_turn(self, winner, win_tile, is_self, about_kong, wall_last,
                 discarder):
        """Build the terminal HU turn (no request)."""
        hand = list(self.hands[winner])
        if is_self:
            hand.remove(win_tile)
        is_4th = self._is_4th_tile(win_tile, winner)
        fan, fan_cnt = self._fan_display(winner, win_tile, hand, self.packs[winner],
                                         is_self, about_kong, wall_last, is_4th)
        if is_self:
            score = self._score_zimo(winner, fan_cnt)
        else:
            score = self._score_ron(winner, discarder, fan_cnt)
        display = {"action": "HU", "player": winner, "fan": fan,
                   "fanCnt": fan_cnt, "score": score, "canHu": [-4] * 4,
                   "tileCnt": list(self.remain)}
        return self._turn({}, display)

    def _is_4th_tile(self, win_tile, winner):
        """True iff the other three copies of the winning tile are already
        exposed on the table (discards + melds of all seats)."""
        vis = 0
        for s in range(4):
            vis += self.discards[s].count(win_tile)
            for m in self.packs[s]:
                vis += meld_tiles((m[0], m[1], m[2])).count(win_tile)
        # the winner holds the winning tile itself; for a ron the discard is
        # already counted in vis, for a zimo it is not.
        return vis == 3

    # ----------------------------------------------------------- arbitration
    def _arbitrate(self, resp, disc_seat):
        claims = {}
        for ss in range(4):
            if ss == disc_seat:
                continue
            parts = (resp.get(str(ss)) or "PASS").split()
            if parts and parts[0] != "PASS":
                claims[ss] = parts
        if not claims:
            return ("PASS", None)
        best = max(_PRIO.get(c[0], 0) for c in claims.values())
        winners = [ss for ss, c in claims.items() if _PRIO.get(c[0], 0) == best]
        if len(winners) > 1:  # multiple HU on one tile: nearest downstream wins
            winners.sort(key=lambda ss: (ss - disc_seat) % 4)
        ws = winners[0]
        c = claims[ws]
        a = c[0]
        if a == "HU":
            return ("HU", ws)
        if a == "PENG":
            return ("PENG", (ws, c[1]))
        if a == "GANG":
            return ("GANG", ws)
        if a == "CHI":
            return ("CHI", (ws, c[1], c[2]))
        return ("PASS", None)

    # ---------------------------------------------------------------- driver
    def _play(self, wall, quan, srand):
        self.quan = quan
        segs = [wall[34 * s:34 * s + 34] for s in range(4)]
        self.segs = segs
        self.remain = [21, 21, 21, 21]
        self.hands = [None] * 4
        self.packs = [[] for _ in range(4)]        # (kind, tile, offer)
        self.discards = [[] for _ in range(4)]

        # ---- INIT ----
        init_req = {str(s): "0 %d %d" % (s, quan) for s in range(4)}
        init_disp = {"action": "INIT", "quan": quan, "srand": srand,
                     "canHu": [0] * 4, "tileCnt": list(self.remain)}
        yield self._turn(init_req, init_disp)

        # ---- DEAL ----
        deal_hand = []
        for s in range(4):
            self.hands[s] = [segs[s][33 - i] for i in range(13)]
            deal_hand.append(list(self.hands[s]))
        deal_req = {str(s): "1 0 0 0 0 " + " ".join(self.hands[s])
                    for s in range(4)}
        deal_disp = {"action": "DEAL", "hand": deal_hand,
                     "canHu": [-4] * 4, "tileCnt": list(self.remain)}
        yield self._turn(deal_req, deal_disp)

        turn = 0        # seat that must draw
        kong = False    # this draw is a kong replacement (about-kong)

        while True:
            # ---------- DRAW phase ----------
            if self.remain[turn] == 0:
                huang = {"action": "HUANG", "canHu": [-4] * 4,
                         "tileCnt": list(self.remain)}
                yield self._turn({}, huang)
                return
            t = segs[turn][self.remain[turn] - 1]
            self.remain[turn] -= 1
            self.hands[turn].append(t)
            req = {str(turn): "2 " + t}
            for i in (1, 2, 3):
                req[str((turn + i) % 4)] = "3 %d DRAW" % turn
            resp = yield self._turn(req, self._draw_display(turn, t, kong))

            r = (resp.get(str(turn)) or "").split()
            act = r[0].upper() if r else "PLAY"

            if act == "HU":                                    # zimo
                wall_last = self.remain[(turn + 1) % 4] == 0
                yield self._hu_turn(turn, t, True, kong, wall_last, None)
                return

            if act == "GANG":                                  # concealed AnGang
                gt = r[1]
                for _ in range(4):
                    self.hands[turn].remove(gt)
                self.packs[turn].append(("GANG", gt, 0))
                greq = {str((turn + i) % 4): "3 %d GANG" % turn for i in range(4)}
                gdisp = {"action": "GANG", "player": turn, "tile": gt,
                         "canHu": [-4] * 4, "tileCnt": list(self.remain)}
                yield self._turn(greq, gdisp)
                kong = True
                continue                                        # same seat redraws

            if act == "BUGANG":                                # added kong
                bt = r[1]
                self.hands[turn].remove(bt)
                for i, m in enumerate(self.packs[turn]):
                    if m[0] == "PENG" and m[1] == bt:
                        self.packs[turn][i] = ("GANG", bt, m[2])
                        break
                breq = {str((turn + i) % 4): "3 %d BUGANG %s" % (turn, bt)
                        for i in range(4)}
                bdisp = {"action": "BUGANG", "player": turn, "tile": bt,
                         "canHu": self._canhu_bugang(turn, bt),
                         "tileCnt": list(self.remain)}
                resp = yield self._turn(breq, bdisp)
                # qianggang arbitration (robbing the kong)
                hus = [ss for ss in range(4) if ss != turn
                       and (resp.get(str(ss)) or "PASS").split()[:1] == ["HU"]]
                if hus:
                    hus.sort(key=lambda ss: (ss - turn) % 4)
                    w = hus[0]
                    wall_last = self.remain[(turn + 1) % 4] == 0
                    yield self._hu_turn(w, bt, False, True, wall_last, turn)
                    return
                kong = True
                continue                                        # replacement draw

            # ---------- PLAY (discard) + claim-resolution chain ----------
            out = r[1]
            self.hands[turn].remove(out)
            self.discards[turn].append(out)
            disc_seat = turn
            preq = {str((turn + i) % 4): "3 %d PLAY %s" % (turn, out)
                    for i in range(4)}
            pdisp = {"action": "PLAY", "player": turn, "tile": out,
                     "canHu": self._canhu_discard(turn, out),
                     "tileCnt": list(self.remain)}
            resp = yield self._turn(preq, pdisp)

            while True:
                kind, data = self._arbitrate(resp, disc_seat)

                if kind == "HU":                                # ron
                    w = data
                    wall_last = self.remain[(disc_seat + 1) % 4] == 0
                    self.discards[disc_seat].pop()  # discard consumed by the win
                    yield self._hu_turn(w, out, False, False, wall_last, disc_seat)
                    return

                if kind == "PENG":
                    ws, ndiscard = data
                    self.discards[disc_seat].pop()              # claimed tile gone
                    self.hands[ws].remove(out)
                    self.hands[ws].remove(out)
                    self.packs[ws].append(("PENG", out, (disc_seat - ws) % 4))
                    self.hands[ws].remove(ndiscard)
                    self.discards[ws].append(ndiscard)
                    creq = {str((ws + i) % 4): "3 %d PENG %s" % (ws, ndiscard)
                            for i in range(4)}
                    cdisp = {"action": "PENG", "player": ws, "tile": ndiscard,
                             "canHu": self._canhu_discard(ws, ndiscard),
                             "tileCnt": list(self.remain)}
                    resp = yield self._turn(creq, cdisp)
                    disc_seat, out = ws, ndiscard
                    continue

                if kind == "GANG":                              # melded kong
                    ws = data
                    self.discards[disc_seat].pop()
                    for _ in range(3):
                        self.hands[ws].remove(out)
                    self.packs[ws].append(("GANG", out, (disc_seat - ws) % 4))
                    greq = {str((ws + i) % 4): "3 %d GANG" % ws for i in range(4)}
                    gdisp = {"action": "GANG", "player": ws, "tile": out,
                             "canHu": [-4] * 4, "tileCnt": list(self.remain)}
                    yield self._turn(greq, gdisp)
                    turn, kong = ws, True
                    break                                       # draw replacement

                if kind == "CHI":
                    ws, mid, ndiscard = data
                    self.discards[disc_seat].pop()
                    suit, n = mid[0], int(mid[1])
                    for x in ("%s%d" % (suit, n - 1), "%s%d" % (suit, n),
                              "%s%d" % (suit, n + 1)):
                        if x != out:
                            self.hands[ws].remove(x)
                    offer = int(out[1]) - n + 2
                    self.packs[ws].append(("CHI", mid, offer))
                    self.hands[ws].remove(ndiscard)
                    self.discards[ws].append(ndiscard)
                    creq = {str((ws + i) % 4): "3 %d CHI %s %s" % (ws, mid, ndiscard)
                            for i in range(4)}
                    cdisp = {"action": "CHI", "player": ws, "tile": ndiscard,
                             "tileCHI": mid,
                             "canHu": self._canhu_discard(ws, ndiscard),
                             "tileCnt": list(self.remain)}
                    resp = yield self._turn(creq, cdisp)
                    disc_seat, out = ws, ndiscard
                    continue

                # no claim: rotate to the next seat's draw
                turn, kong = (disc_seat + 1) % 4, False
                break
