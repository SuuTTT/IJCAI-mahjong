#!/usr/bin/env python3
"""validate_engine.py — correctness validator for Chinese Standard Mahjong (MCR) engines
against the IJCAI-2026 Final Stage-2 golden game set (Dannibal/mcr-final2026-testset).

Two modes:

1. Self-test (validates the dataset itself against the reference replay semantics):

       python3 validate_engine.py --self-test mcr_final2026_full.jsonl.gz [--jobs 64]

   Replays every game with a pure-python reference checker (the corrected
   "replay_harness2" semantics: PENG/CHI-embedded discards, live-discard GANG
   discrimination, BUGANG handling, ron/zimo/qianggang terminal labeling) and
   asserts, for every game:
     - wall integrity (136 tiles, 4 of each of the 34 kinds)
     - deal & draw order: seat s is dealt/draws from wall[34*s .. 34*s+33],
       consumed from the BACK (index 34*s+33 first); tileCnt bookkeeping
     - per-step hand/meld legality (every discard from hand, PENG needs 2 in
       hand, CHI only from the left neighbour with both fillers in hand,
       melded GANG needs 3, AnGang needs 4 after own draw, BUGANG upgrades an
       existing PENG)
     - claim resolution priority (HU > PENG/GANG > CHI) and actor rotation
     - response<->event consistency (the granted claim's response string must
       match the following judge event, including the embedded discard)
     - terminal: ending classification (zimo/ron/draw + qianggang), fan_total
       == sum(value*cnt) of the fan list, exact score arithmetic
       (zimo: winner +3*(8+fan), others -(8+fan);
        ron:  winner +(3*8+fan), discarder -(8+fan), bystanders -8),
       scores sum to 0, and agreement with the record's `expected` block.

2. Engine mode (validates YOUR engine — the product):

       python3 validate_engine.py --engine mymodule:MyEngine mcr_final2026_golden.jsonl

   Your engine must implement the interface below. The validator drives it
   with the logged bot responses turn by turn and requires the engine to
   reproduce the judge's protocol stream exactly: every per-seat request
   string, every display event (claim resolutions, gang discrimination, win
   detection) and the terminal fan/score result.

       class MyEngine:
           def reset(self, wall: list[str], quan: int, srand: int) -> dict:
               '''Start a game. wall = 136 tiles; seat s draws from
               wall[34*s : 34*s+34], back to front. Returns the first turn:
               {"requests": {"0": "0 0 0", ...}, "display": {"action": "INIT", ...}}'''
           def step(self, responses: dict[str, str]) -> dict:
               '''Feed the four seats' response strings to the pending
               requests; return the next turn dict:
               {"requests": {...} or {}, "display": {...}}.
               The terminal turn's display has action HU (with fan/fanCnt/
               score) or HUANG.'''

   By default `canHu` and `tileCnt` inside displays are compared too (they
   are exactly reproducible); use --loose to compare only
   action/player/tile/tileCHI/hand/fan/fanCnt/score.

   `--demo` runs engine mode with a built-in engine that replays the stored
   turns (interface smoke test).

Dataset: https://huggingface.co/datasets/Dannibal/mcr-final2026-testset
"""
import argparse
import gzip
import json
import sys
from collections import Counter

TILE_KINDS = ([c + str(n) for c in "WBT" for n in range(1, 10)]
              + ["F" + str(n) for n in range(1, 5)]
              + ["J" + str(n) for n in range(1, 4)])

CLAIM_PRIO = {"HU": 3, "GANG": 2, "PENG": 2, "CHI": 1}


class CheckFail(Exception):
    pass


def _ck(cond, msg):
    if not cond:
        raise CheckFail(msg)


def classify_terminal(displays):
    """Harness2 terminal semantics from the ordered display stream.
    Returns dict(ending, winner, discarder, qianggang) — ending in
    zimo|ron|draw. Ron off a PENG/CHI-embedded discard counts the claimer
    as discarder; HU immediately after BUGANG is qianggang (robbed kong),
    discarder = the BUGANG player."""
    live = None          # (tile, seat) of the unclaimed discard
    prev = None
    for d in displays:
        a = d["action"]
        if a == "DRAW":
            live = None
        elif a in ("PLAY", "PENG", "CHI"):
            live = (d["tile"], d["player"])
        elif a == "GANG":
            live = None
        elif a == "BUGANG":
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


def expected_scores(ending, winner, discarder, fan_total):
    if ending == "draw":
        return [0, 0, 0, 0]
    sc = [0, 0, 0, 0]
    if ending == "zimo":
        for s in range(4):
            sc[s] = 3 * (8 + fan_total) if s == winner else -(8 + fan_total)
    else:  # ron (incl. qianggang)
        for s in range(4):
            if s == winner:
                sc[s] = 3 * 8 + fan_total
            elif s == discarder:
                sc[s] = -(8 + fan_total)
            else:
                sc[s] = -8
    return sc


def check_record(rec):
    """Full structural + semantic check of one dataset record.
    Raises CheckFail on the first violation. Returns a dict of facts
    (event counts etc.) usable for tagging."""
    wall = rec["wall"]
    _ck(len(wall) == 136, "wall length %d != 136" % len(wall))
    _ck(Counter(wall) == Counter({t: 4 for t in TILE_KINDS}),
        "wall is not 4x each of the 34 tile kinds")
    segs = [wall[34 * s:34 * s + 34] for s in range(4)]
    remain = [34, 34, 34, 34]      # tiles left in each seat's segment
    hands = [None] * 4
    packs = [[] for _ in range(4)]  # (type, tile, offer_seat)
    live = None                     # (tile, seat) live unclaimed discard
    prev_a = prev_p = None
    turns = rec["turns"]
    facts = Counter()
    displays = [t["display"] for t in turns]

    def hand_ok(s):
        _ck(len(hands[s]) == 13 - 3 * len(packs[s]),
            "seat %d hand size %d with %d melds" % (s, len(hands[s]), len(packs[s])))

    for k, turn in enumerate(turns):
        d = turn["display"]
        a = d["action"]
        nxt = turns[k + 1]["display"] if k + 1 < len(turns) else None
        resp = turn.get("responses") or {}

        if a == "INIT":
            _ck(d.get("quan") == rec["quan"], "INIT quan mismatch")
            _ck(d.get("srand") == rec["srand"], "INIT srand mismatch")
        elif a == "DEAL":
            for s in range(4):
                expect = [segs[s][33 - i] for i in range(13)]
                _ck(d["hand"][s] == expect,
                    "seat %d dealt hand != last-13-reversed of wall segment" % s)
                hands[s] = list(d["hand"][s])
                remain[s] = 21
        elif a == "DRAW":
            s, t = d["player"], d["tile"]
            _ck(remain[s] > 0, "seat %d draws from empty segment" % s)
            _ck(segs[s][remain[s] - 1] == t,
                "draw %s != wall order tile %s" % (t, segs[s][remain[s] - 1]))
            remain[s] -= 1
            _ck(d["tileCnt"][s] == remain[s], "tileCnt mismatch after draw")
            # rotation: replacement draw after own kong, else next seat
            if prev_a in ("GANG", "BUGANG"):
                _ck(s == prev_p, "replacement draw by wrong seat")
            elif prev_a in ("PLAY", "PENG", "CHI"):
                _ck(s == (prev_p + 1) % 4, "draw out of rotation")
            elif prev_a == "DEAL":
                _ck(s == 0, "first draw not by dealer (seat 0)")
            live = None
            hands[s].append(t)
            # drawer's own response drives the next event
            r = (resp.get(str(s)) or "").split()
            if r:
                if r[0] == "PLAY":
                    _ck(nxt and nxt["action"] == "PLAY" and nxt["player"] == s
                        and nxt["tile"] == r[1], "PLAY response not honored")
                elif r[0] == "GANG":
                    _ck(nxt and nxt["action"] == "GANG" and nxt["player"] == s
                        and nxt.get("tile") == r[1], "AnGang response not honored")
                elif r[0] == "BUGANG":
                    _ck(nxt and nxt["action"] == "BUGANG" and nxt["player"] == s
                        and nxt["tile"] == r[1], "BuGang response not honored")
                elif r[0] == "HU":
                    _ck(nxt and nxt["action"] == "HU" and nxt["player"] == s,
                        "zimo HU response not honored")
        elif a in ("PLAY", "PENG", "CHI"):
            s = d["player"]
            out = d["tile"]
            if a == "PLAY":
                _ck(prev_a == "DRAW" and prev_p == s, "PLAY not after own draw")
            elif a == "PENG":
                _ck(live is not None and live[1] != s, "PENG with no live discard")
                ct = live[0]
                _ck(hands[s].count(ct) >= 2, "PENG without 2 matching tiles in hand")
                hands[s].remove(ct); hands[s].remove(ct)
                packs[s].append(("PENG", ct, live[1]))
                facts["minggang_or_peng"] += 1
                facts["peng"] += 1
            else:  # CHI
                _ck(live is not None, "CHI with no live discard")
                ct, ds = live
                _ck((s - ds) % 4 == 1, "CHI from non-left neighbour")
                mid = d["tileCHI"]
                _ck(ct[0] == mid[0] and mid[0] in "WBT", "CHI suit mismatch")
                m, c = int(mid[1]), int(ct[1])
                _ck(c in (m - 1, m, m + 1), "claimed tile not part of chi run")
                need = [mid[0] + str(x) for x in (m - 1, m, m + 1) if x != c]
                for tt in need:
                    _ck(tt in hands[s], "CHI filler %s not in hand" % tt)
                    hands[s].remove(tt)
                packs[s].append(("CHI", mid, ds))
                facts["chi"] += 1
            # the embedded (PENG/CHI) or plain discard
            _ck(out in hands[s], "discard %s not in hand" % out)
            hands[s].remove(out)
            hand_ok(s)
            live = (out, s)
            # claim arbitration on this discard
            claims = {}
            for ss in range(4):
                if ss == s:
                    continue
                rr = (resp.get(str(ss)) or "PASS").split()
                if rr and rr[0] != "PASS":
                    claims[ss] = rr
            if claims:
                best = max(CLAIM_PRIO.get(rr[0], 0) for rr in claims.values())
                winners = [ss for ss, rr in claims.items()
                           if CLAIM_PRIO.get(rr[0], 0) == best]
                if len(winners) > 1:  # multiple HU on one tile: nearest downstream
                    facts["multi_hu"] += 1
                    winners = sorted(winners, key=lambda ss: (ss - s) % 4)[:1]
                if len(claims) > 1:
                    facts["multi_claim"] += 1
                ws = winners[0]
                rr = claims[ws]
                _ck(nxt is not None, "claim pending but log ended")
                _ck(nxt["player"] == ws,
                    "claim priority: expected seat %d to act next" % ws)
                _ck({"HU": "HU", "PENG": "PENG", "CHI": "CHI",
                     "GANG": "GANG"}[rr[0]] == nxt["action"],
                    "granted claim %s != next event %s" % (rr[0], nxt["action"]))
                if rr[0] == "PENG":
                    _ck(nxt["tile"] == rr[1], "PENG follow-up discard mismatch")
                elif rr[0] == "CHI":
                    _ck(nxt["tileCHI"] == rr[1] and nxt["tile"] == rr[2],
                        "CHI mid/follow-up mismatch")
            else:
                _ck(nxt is not None and nxt["action"] in ("DRAW", "HUANG"),
                    "no claim: expected DRAW/HUANG next, got %s"
                    % (nxt and nxt["action"]))
        elif a == "GANG":
            s = d["player"]
            t = d.get("tile")
            if live is not None:              # melded gang of the live discard
                ct, ds = live
                _ck(s != ds, "GANG own discard")
                _ck(t == ct, "melded GANG tile %s != live discard %s" % (t, ct))
                _ck(hands[s].count(ct) == 3, "melded GANG without 3 in hand")
                for _ in range(3):
                    hands[s].remove(ct)
                packs[s].append(("GANG", ct, ds))
                live = None
                facts["minggang"] += 1
            else:                             # concealed kong after own draw
                _ck(prev_a == "DRAW" and prev_p == s, "AnGang not after own draw")
                _ck(hands[s].count(t) == 4, "AnGang without 4 in hand")
                for _ in range(4):
                    hands[s].remove(t)
                packs[s].append(("ANGANG", t, s))
                facts["angang"] += 1
            _ck(nxt is not None and ((nxt["action"] == "DRAW" and nxt["player"] == s)
                                     or nxt["action"] == "HUANG"),
                "no replacement draw after GANG")
        elif a == "BUGANG":
            s, t = d["player"], d["tile"]
            up = [p for p in packs[s] if p[0] == "PENG" and p[1] == t]
            _ck(up, "BUGANG without matching PENG meld")
            _ck(t in hands[s], "BUGANG tile not in hand")
            hands[s].remove(t)
            packs[s][packs[s].index(up[0])] = ("GANG", t, up[0][2])
            live = None
            facts["bugang"] += 1
            # qianggang arbitration
            hus = [ss for ss in range(4) if ss != s
                   and (resp.get(str(ss)) or "PASS").split()[:1] == ["HU"]]
            if hus:
                _ck(nxt is not None and nxt["action"] == "HU"
                    and nxt["player"] in hus, "qianggang HU not honored")
            else:
                _ck(nxt is not None and ((nxt["action"] == "DRAW"
                                          and nxt["player"] == s)
                                         or nxt["action"] == "HUANG"),
                    "no replacement draw after BUGANG")
        elif a == "HUANG":
            nxt_drawer = (prev_p + 1) % 4 if prev_a in ("PLAY", "PENG", "CHI") \
                else prev_p
            _ck(remain[nxt_drawer] == 0,
                "HUANG but next drawer still has %d wall tiles" % remain[nxt_drawer])
        elif a == "HU":
            pass                              # terminal block checked below
        else:
            raise CheckFail("unknown display action %r" % a)
        prev_a, prev_p = a, d.get("player", prev_p)

    # ---- terminal / expected block ----
    exp = rec["expected"]
    cls = classify_terminal(displays)
    _ck(cls["ending"] == exp["ending"], "ending %s != expected %s"
        % (cls["ending"], exp["ending"]))
    _ck(cls["winner"] == exp["winner"], "winner mismatch")
    _ck(cls["discarder"] == exp["discarder"], "discarder mismatch")
    _ck(cls["qianggang"] == exp["qianggang"], "qianggang flag mismatch")
    last = displays[-1]
    if exp["ending"] == "draw":
        _ck(last["action"] == "HUANG", "draw game does not end with HUANG")
        _ck(exp["scores"] == [0, 0, 0, 0], "draw scores not all 0")
        _ck(exp["fan_total"] == 0 and exp["fan"] == [], "draw with fan")
    else:
        _ck(last["action"] == "HU", "hu game does not end with HU")
        fan_sum = sum(v * c for _, v, c in exp["fan"])
        _ck(fan_sum == exp["fan_total"],
            "fan list sums to %d != fan_total %d" % (fan_sum, exp["fan_total"]))
        _ck(last["fanCnt"] == exp["fan_total"], "display fanCnt != fan_total")
        _ck(list(last["score"]) == list(exp["scores"]), "display score != expected")
        want = expected_scores(exp["ending"], exp["winner"], exp["discarder"],
                               exp["fan_total"])
        _ck(want == list(exp["scores"]),
            "score arithmetic: computed %s != logged %s" % (want, exp["scores"]))
        _ck(exp["fan_total"] >= 8, "winning hand below the 8-fan minimum")
    _ck(sum(exp["scores"]) == 0, "scores do not sum to 0")
    facts[exp["ending"]] += 1
    return dict(facts)


# ---------------------------------------------------------------- engine mode
LOOSE_KEYS = ("action", "player", "tile", "tileCHI", "hand", "fan", "fanCnt", "score")


def _disp_eq(a, b, loose):
    if loose:
        return all(a.get(k) == b.get(k) for k in LOOSE_KEYS)
    ka = {k: v for k, v in a.items() if v is not None}
    kb = {k: v for k, v in b.items() if v is not None}
    return ka == kb


class DemoReplayEngine:
    """Interface demo: 'engine' that replays the record's own stored turns."""

    def bind(self, rec):
        self._turns = rec["turns"]
        self._i = 0

    def reset(self, wall, quan, srand):
        self._i = 0
        return self._emit()

    def step(self, responses):
        return self._emit()

    def _emit(self):
        t = self._turns[self._i]
        self._i += 1
        return {"requests": t.get("request") or {}, "display": t["display"]}


def run_engine(engine, rec, loose=False):
    """Drive `engine` through one game; raise CheckFail on first divergence."""
    turns = rec["turns"]
    out = engine.reset(list(rec["wall"]), rec["quan"], rec["srand"])
    for k, turn in enumerate(turns):
        _ck(_disp_eq(out["display"], turn["display"], loose),
            "turn %d display diverged:\n engine: %s\n oracle: %s"
            % (k, json.dumps(out["display"], ensure_ascii=False),
               json.dumps(turn["display"], ensure_ascii=False)))
        if turn.get("request"):
            _ck(out.get("requests") == turn["request"],
                "turn %d request strings diverged" % k)
        if k + 1 < len(turns):
            out = engine.step(dict(turn.get("responses") or {}))


# --------------------------------------------------------------------- driver
def iter_records(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _self_one(line):
    rec = json.loads(line)
    try:
        check_record(rec)
        return (rec["game_id"], None, rec.get("tags", []))
    except CheckFail as e:
        return (rec["game_id"], str(e), rec.get("tags", []))
    except Exception as e:  # noqa: BLE001
        return (rec["game_id"], "internal: %r" % e, rec.get("tags", []))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", help=".jsonl or .jsonl.gz test file")
    ap.add_argument("--self-test", action="store_true",
                    help="validate the dataset against the reference replay")
    ap.add_argument("--engine", metavar="module:Class",
                    help="validate an engine implementation")
    ap.add_argument("--demo", action="store_true",
                    help="engine mode with the built-in replay engine")
    ap.add_argument("--loose", action="store_true",
                    help="engine mode: compare only core display keys")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.self_test:
        op = gzip.open if args.dataset.endswith(".gz") else open
        with op(args.dataset, "rt", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        if args.limit:
            lines = lines[:args.limit]
        if args.jobs > 1:
            import multiprocessing as mp
            with mp.Pool(args.jobs) as pool:
                results = pool.map(_self_one, lines, chunksize=16)
        else:
            results = [_self_one(ln) for ln in lines]
        fails = [(g, e) for g, e, _ in results if e]
        n_disc = sum(1 for _, _, tags in results if "gb_discrepancy" in tags)
        print("self-test: %d/%d games PASS (%d known-discrepancy games included"
              " — they pass: the judge is the ground truth)"
              % (len(results) - len(fails), len(results), n_disc))
        for g, e in fails[:20]:
            print("FAIL %s: %s" % (g, e))
        sys.exit(1 if fails else 0)

    if args.demo or args.engine:
        if args.demo:
            eng = DemoReplayEngine()
        else:
            modname, clsname = args.engine.split(":")
            mod = __import__(modname)
            eng = getattr(mod, clsname)()
        n = ok = 0
        for rec in iter_records(args.dataset):
            n += 1
            if args.limit and n > args.limit:
                n -= 1
                break
            if args.demo:
                eng.bind(rec)
            try:
                run_engine(eng, rec, loose=args.loose)
                ok += 1
            except CheckFail as e:
                print("FAIL %s: %s" % (rec["game_id"], e))
        print("engine mode: %d/%d games reproduced exactly" % (ok, n))
        sys.exit(0 if ok == n else 1)

    ap.error("choose --self-test, --engine or --demo")


if __name__ == "__main__":
    main()
