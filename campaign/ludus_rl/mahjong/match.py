"""Match driver over the ORACLE-VALIDATED steppable engine (``MyEngine``).

``mahjong.engine.play_game`` is convenient but NOT byte-exact with the official
IJCAI judge, so protocol-sensitive agents (the kdens3 champion) desync when
driven through it.  ``mahjong.validate_adapter.MyEngine`` *is* byte-exact (it
reproduced the full 12,288-game oracle), and it is steppable:

    turn = eng.reset(wall, quan, srand)      # -> {"requests": {...}, "display": {...}}
    turn = eng.step({seat_str: response, ...})

``run_match`` drives ``MyEngine`` with a list of four Botzone-protocol agents
(objects exposing ``respond(request_str) -> response_str``), reconstructs the
full-information viewer frames from the judge display stream (mirroring the
validator's ``check_record`` bookkeeping), and returns the terminal result.

Because ``MyEngine`` emits judge-exact requests, the champion sees exactly what
it saw on Botzone and behaves correctly (no illegal moves / desyncs).
"""

from mahjong.validate_adapter import MyEngine


# --------------------------------------------------------------------------
# terminal classification (copied from validate_engine.py:classify_terminal so
# match.py has no runtime dependency on the test harness)
# --------------------------------------------------------------------------
def classify_terminal(displays):
    """Ending semantics from the ordered display stream.

    Returns ``dict(ending, winner, discarder, qianggang)`` with ending in
    ``zimo|ron|draw``.  Ron off a PENG/CHI-embedded discard counts the claimer
    as discarder; HU immediately after BUGANG is qianggang (robbed kong).
    """
    live = None
    prev = None
    for d in displays:
        a = d["action"]
        if a == "DRAW":
            live = None
        elif a in ("PLAY", "PENG", "CHI"):
            live = (d["tile"], d["player"])
        elif a in ("GANG", "BUGANG"):
            live = None
        elif a == "HUANG":
            return dict(ending="draw", winner=None, discarder=None, qianggang=False)
        elif a == "HU":
            w = d["player"]
            if prev is not None and prev["action"] == "BUGANG":
                return dict(ending="ron", winner=w,
                            discarder=prev["player"], qianggang=True)
            if live is not None:
                return dict(ending="ron", winner=w, discarder=live[1], qianggang=False)
            return dict(ending="zimo", winner=w, discarder=None, qianggang=False)
        prev = d
    return dict(ending="unknown", winner=None, discarder=None, qianggang=False)


# --------------------------------------------------------------------------
# board-state reconstruction -> viewer frames
# --------------------------------------------------------------------------
_DIR = ["East", "South", "West", "North"]
_SUIT_CN = {"W": "万", "T": "条", "B": "筒"}
_HONOR_CN = {"F1": "东", "F2": "南", "F3": "西", "F4": "北",
             "J1": "中", "J2": "发", "J3": "白"}


def _tile_human(code):
    if code in _HONOR_CN:
        return _HONOR_CN[code]
    return code[1:] + _SUIT_CN.get(code[0], "")


class _Board:
    """Replays the display stream to rebuild full-info board snapshots.

    Mirrors ``validate_engine.check_record``'s bookkeeping exactly: deal =
    last-13-reversed of each 34-tile segment, per-seat wall counter starting at
    21, claim-consumed discards popped from the discarder's pile.
    """

    def __init__(self):
        self.hands = [[], [], [], []]
        self.packs = [[], [], [], []]           # (kind, tile, offer)
        self.discards = [[], [], [], []]
        self.remain = [21, 21, 21, 21]
        self.live = None                        # (tile, seat) unclaimed discard
        self.prev_a = None
        self.prev_p = None
        self.last_draw = None                   # last self-drawn tile
        self.prev_bugang = None                 # last BUGANG tile
        self.frames = []

    def _snap(self, turn, mode, last, scores=None):
        fr = {
            "hands": [sorted(h) for h in self.hands],
            "melds": [[list(m) for m in ms] for ms in self.packs],
            "discards": [list(d) for d in self.discards],
            "turn": turn, "mode": mode, "last": last,
            "wall_left": list(self.remain),
        }
        if scores is not None:
            fr["scores"] = scores
        return fr

    def apply(self, d):
        """Process one display event; append a frame for renderable events."""
        a = d["action"]
        if a == "INIT":
            pass
        elif a == "DEAL":
            for s in range(4):
                self.hands[s] = list(d["hand"][s])
                self.remain[s] = 21
            self.frames.append(self._snap(0, "draw", ""))
        elif a == "DRAW":
            s, t = d["player"], d["tile"]
            self.remain[s] -= 1
            self.hands[s].append(t)
            self.live = None
            self.last_draw = t
            self.frames.append(self._snap(s, "draw", "DRAW %d %s" % (s, t)))
        elif a == "PLAY":
            s, out = d["player"], d["tile"]
            self.hands[s].remove(out)
            self.discards[s].append(out)
            self.live = (out, s)
            self.frames.append(self._snap(s, "resolve", "3 %d PLAY %s" % (s, out)))
        elif a == "PENG":
            s, out = d["player"], d["tile"]
            ct, ds = self.live
            self._pop_live()
            self.hands[s].remove(ct)
            self.hands[s].remove(ct)
            self.packs[s].append(["PENG", ct, (ds - s) % 4])
            self.hands[s].remove(out)
            self.discards[s].append(out)
            self.live = (out, s)
            self.frames.append(self._snap(s, "resolve", "3 %d PENG %s" % (s, out)))
        elif a == "CHI":
            s, out, mid = d["player"], d["tile"], d["tileCHI"]
            ct, ds = self.live
            self._pop_live()
            suit, n = mid[0], int(mid[1])
            for x in ("%s%d" % (suit, n - 1), "%s%d" % (suit, n),
                      "%s%d" % (suit, n + 1)):
                if x != ct:
                    self.hands[s].remove(x)
            offer = int(ct[1]) - n + 2
            self.packs[s].append(["CHI", mid, offer])
            self.hands[s].remove(out)
            self.discards[s].append(out)
            self.live = (out, s)
            self.frames.append(self._snap(s, "resolve",
                                          "3 %d CHI %s %s" % (s, mid, out)))
        elif a == "GANG":
            s, t = d["player"], d["tile"]
            if self.live is not None and self.prev_a in ("PLAY", "PENG", "CHI"):
                ct, ds = self.live                 # melded gang of live discard
                self._pop_live()
                for _ in range(3):
                    self.hands[s].remove(ct)
                self.packs[s].append(["GANG", ct, (ds - s) % 4])
            else:                                  # concealed (an)gang after draw
                for _ in range(4):
                    self.hands[s].remove(t)
                self.packs[s].append(["GANG", t, 0])
            self.live = None
            self.frames.append(self._snap(s, "draw", "3 %d GANG" % s))
        elif a == "BUGANG":
            s, t = d["player"], d["tile"]
            self.hands[s].remove(t)
            for m in self.packs[s]:
                if m[0] == "PENG" and m[1] == t:
                    m[0] = "GANG"
                    break
            self.live = None
            self.prev_bugang = t
            self.frames.append(self._snap(s, "draw", "3 %d BUGANG %s" % (s, t)))
        elif a == "HU":
            w = d["player"]
            fan = d.get("fanCnt")
            if self.prev_a == "BUGANG":
                how, wtile = "robkong", self.prev_bugang
            elif self.live is not None:
                how, wtile = "rong", self.live[0]
                self._pop_live()                   # discard consumed by the win
            else:
                how, wtile = "zimo", self.last_draw
            score = d.get("score") or [0, 0, 0, 0]
            scores = {i: score[i] for i in range(4)}
            self.frames.append(self._snap(
                w, how, "HU %d %s %s fan=%s" % (w, wtile, how, fan), scores))
        elif a == "HUANG":
            scores = {i: 0 for i in range(4)}
            self.frames.append(self._snap(-1, "draw", "HUANG", scores))
        self.prev_a = a
        self.prev_p = d.get("player", self.prev_p)

    def _pop_live(self):
        if self.live is not None:
            tile, seat = self.live
            if self.discards[seat] and self.discards[seat][-1] == tile:
                self.discards[seat].pop()


# --------------------------------------------------------------------------
# verbose fan breakdown for the win banner
# --------------------------------------------------------------------------
def _winning_fan_list(displays, board, quan):
    """Per-fan breakdown for the terminal HU, as ``[{cn, en, value, cnt, desc}]``.

    The oracle HU display already carries the Chinese fan list (``value``/
    ``cnt``); here we recover the ENGLISH names too by reconstructing the
    winner's concealed hand + melds + winning tile from the final board state
    and calling :func:`calc_fan_raw` in verbose mode (4-tuples
    ``(value, cnt, cn, en)``, exactly what the Botzone judge uses).  The
    reconstruction is cross-checked against the oracle ``fanCnt``; on any
    mismatch it falls back to the display's authoritative cn list with English
    names pulled from the glossary.  Returns ``None`` for a non-HU terminal.
    """
    from mahjong.engine import calc_fan_raw, meld_tiles
    from mahjong import fan_glossary

    last = displays[-1]
    if last.get("action") != "HU":
        return None
    w = last["player"]
    d2 = displays[-2] if len(displays) >= 2 else {}
    a2 = d2.get("action")
    if a2 == "BUGANG":                                   # robbing the kong
        is_self, about_kong, wtile, pivot = False, True, d2["tile"], d2["player"]
    elif a2 in ("PLAY", "PENG", "CHI"):                  # ron off a discard
        is_self, about_kong, wtile, pivot = False, False, d2["tile"], d2["player"]
    else:                                                # DRAW -> self-draw (zimo)
        d3 = displays[-3] if len(displays) >= 3 else {}
        is_self, wtile, pivot = True, d2.get("tile"), w
        about_kong = (d3.get("action") in ("GANG", "BUGANG")
                      and d3.get("player") == w)         # kong-replacement draw
    tile_cnt = last.get("tileCnt") or [0, 0, 0, 0]
    wall_last = tile_cnt[(pivot + 1) % 4] == 0
    concealed = list(board.hands[w])
    if is_self and wtile in concealed:
        concealed.remove(wtile)                          # win tile is drawn, not a set
    vis = 0                                              # copies already on the table
    for s in range(4):
        vis += board.discards[s].count(wtile)
        for m in board.packs[s]:
            vis += meld_tiles((m[0], m[1], m[2])).count(wtile)
    is_4th = (vis == 3)
    packs = tuple((m[0], m[1], m[2]) for m in board.packs[w])

    entries = None
    try:
        raw = calc_fan_raw(packs, tuple(concealed), wtile, is_self, about_kong,
                           wall_last, is_4th, w, int(quan), verbose=True)
        if sum(v * c for v, c, *_ in raw) == last.get("fanCnt"):
            entries = [(v, c, cn, en) for v, c, cn, en in raw]
    except Exception:
        entries = None
    if entries is None:                                  # fall back to oracle cn list
        entries = [(f["value"], f["cnt"], f["name"], fan_glossary.en_for(f["name"]))
                   for f in (last.get("fan") or [])]
    return [{"cn": cn, "en": en or fan_glossary.en_for(cn),
             "value": v, "cnt": c,
             "desc": fan_glossary.desc_for(cn=cn, en=en)}
            for v, c, cn, en in entries]


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run_match(agents, wall, quan=0, srand=0):
    """Play one MCR game through ``MyEngine`` and return the result + frames.

    ``agents`` is a length-4 list of Botzone-protocol objects (``.respond``).
    Seat ``i`` is served by ``agents[i]``.  Returns::

        {"scores": {0..3: int}, "winner": seat|None, "fan": int|None,
         "ending": "zimo"|"rong"|"draw", "frames": [...]}
    """
    assert len(agents) == 4, "need exactly 4 agents"
    wall = list(wall)
    assert len(wall) == 136, "wall must be 136 tiles"

    eng = MyEngine()
    board = _Board()
    displays = []

    turn = eng.reset(wall, int(quan), int(srand))
    while True:
        disp = turn["display"]
        displays.append(disp)
        board.apply(disp)
        reqs = turn["requests"]
        if not reqs:                       # terminal turn (HU / HUANG)
            break
        responses = {}
        for seat_str, req in reqs.items():
            responses[seat_str] = agents[int(seat_str)].respond(req)
        turn = eng.step(responses)

    cls = classify_terminal(displays)
    ending = {"ron": "rong"}.get(cls["ending"], cls["ending"])
    last = displays[-1]
    fan_list = None
    if ending == "draw":
        scores = {i: 0 for i in range(4)}
        winner = None
        fan = None
    else:
        sc = last.get("score") or [0, 0, 0, 0]
        scores = {i: sc[i] for i in range(4)}
        winner = cls["winner"]
        fan = last.get("fanCnt")
        try:
            fan_list = _winning_fan_list(displays, board, quan)
        except Exception:
            fan_list = None

    return {"scores": scores, "winner": winner, "fan": fan,
            "fan_list": fan_list, "ending": ending, "frames": board.frames}
