#!/usr/bin/env python3
"""build_testset.py — transform the 12,288 IJCAI-2026 Final Stage-2 raw match logs
into the MCR engine-correctness test dataset (Dannibal/mcr-final2026-testset).

Reads  /root/final2_harvest/final2_all.jsonl.gz
Writes /root/final2_harvest/testset/mcr_final2026_full.jsonl.gz   (all games)
       /root/final2_harvest/testset/mcr_final2026_golden.jsonl    (curated subset)
       /root/final2_harvest/testset/TAGS_SUMMARY.json             (counts, grammar, gb scan)

Every record is structurally validated with validate_engine.check_record
(the corrected replay-harness2 semantics) at build time, and every HU ending
is re-scored with the python MahjongGB fan calculator via the caiest
FeatureAgent replay to surface judge-vs-MahjongGB discrepancies.
"""
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict

BASE = "/root/final2_harvest"
OUT = BASE + "/testset"
sys.path.insert(0, OUT)               # validate_engine.py lives here
sys.path.insert(0, "/root/caiest_repro")

from validate_engine import check_record, classify_terminal, CheckFail  # noqa: E402
from feature import FeatureAgent as CaiAgent  # noqa: E402
from MahjongGB import MahjongFanCalculator  # noqa: E402


class ProbeAgent(CaiAgent):
    """CaiAgent whose 8-fan Hu gate also records the MahjongGB fan calculation.

    Records two variants:
      * cai-style flags exactly as the widely-copied FeatureAgent computes them
        (is4thTile = shownTiles[winTile] == 4 — WRONG for self-drawn wins: the
        just-drawn tile is never in shownTiles, so 和绝张 is unreachable on zimo)
      * corrected is4thTile = shownTiles + (1 if self-drawn) == 4
    """

    def _calc(self, winTile, isSelfDrawn, isAboutKong, is4th):
        fans = MahjongFanCalculator(
            pack=tuple(self.packs[0]), hand=tuple(self.hand), winTile=winTile,
            flowerCount=0, isSelfDrawn=isSelfDrawn,
            is4thTile=is4th, isAboutKong=isAboutKong,
            isWallLast=self.wallLast, seatWind=self.seatWind,
            prevalentWind=self.prevalentWind, verbose=True)
        return (sum(fp * c for fp, c, _, _ in fans),
                [[fn, fp, c] for fp, c, fn, _ in fans])

    def _check_mahjong(self, winTile, isSelfDrawn=False, isAboutKong=False):
        shown = self.shownTiles[winTile]
        cai4 = shown == 4
        fix4 = (shown + (1 if isSelfDrawn else 0)) == 4
        try:
            fc, fl = self._calc(winTile, isSelfDrawn, isAboutKong, cai4)
            if fix4 != cai4:
                fc2, fl2 = self._calc(winTile, isSelfDrawn, isAboutKong, fix4)
            else:
                fc2, fl2 = fc, fl
            self.last_calc = ("ok", fc, fl, fc2, fl2)
            return fc >= 8
        except Exception as e:  # noqa: BLE001
            self.last_calc = ("error", str(e), None, None, None)
            return False


def gb_probe(displays, quan):
    """Replay via cai FeatureAgents; return the winner's MahjongGB calc at HU."""
    agents = None
    live = None
    calc = None
    for d in displays:
        a = d["action"]
        if a == "DEAL":
            agents = [ProbeAgent(s) for s in range(4)]
            for s in range(4):
                agents[s].request2obs("Wind %d" % quan)
                agents[s].request2obs("Deal " + " ".join(d["hand"][s]))
        elif a == "DRAW":
            p = d["player"]
            live = None
            for s in range(4):
                agents[s].request2obs("Draw %s" % d["tile"] if s == p
                                      else "Player %d Draw" % p)
        elif a == "PLAY":
            p, t = d["player"], d["tile"]
            live = t
            for s in range(4):
                agents[s].request2obs("Player %d Play %s" % (p, t))
        elif a == "PENG":
            p, out = d["player"], d["tile"]
            for s in range(4):
                agents[s].request2obs("Player %d Peng" % p)
            live = out
            for s in range(4):
                agents[s].request2obs("Player %d Play %s" % (p, out))
        elif a == "CHI":
            p, out, mid = d["player"], d["tile"], d["tileCHI"]
            for s in range(4):
                agents[s].request2obs("Player %d Chi %s" % (p, mid))
            live = out
            for s in range(4):
                agents[s].request2obs("Player %d Play %s" % (p, out))
        elif a == "GANG":
            p = d["player"]
            if live is not None:
                for s in range(4):
                    agents[s].request2obs("Player %d Gang" % p)
                live = None
            else:
                for s in range(4):
                    agents[s].request2obs("Player %d AnGang %s" % (p, d["tile"]))
        elif a == "BUGANG":
            p = d["player"]
            live = None
            for s in range(4):
                agents[s].request2obs("Player %d BuGang %s" % (p, d["tile"]))
        elif a == "HU":
            w = d["player"]
            calc = getattr(agents[w], "last_calc", None)
    return calc


def build_record(raw):
    logs = raw["logs"]
    init = json.loads(raw["initdata"]) if isinstance(raw["initdata"], str) \
        else raw["initdata"]
    players = []
    for p in raw.get("players", []):
        nm = p.get("name", "")
        players.append(nm.lstrip("[").replace("]", "/", 1) if "]" in nm else nm)

    turns = []
    req_patterns = Counter()
    fin_scores = None
    for i, entry in enumerate(logs):
        if not (isinstance(entry, dict) and "output" in entry):
            continue
        out = entry["output"] or {}
        disp = out.get("display")
        if isinstance(disp, str):
            disp = json.loads(disp)
        if not (isinstance(disp, dict) and disp.get("action")):
            continue
        turn = {"display": disp}
        if out.get("command") == "request":
            turn["request"] = {k: out["content"][k] for k in sorted(out["content"])}
            for k, v in turn["request"].items():
                toks = v.split()
                if toks[0] == "0":
                    req_patterns["0 <seat> <quan>  (game start)"] += 1
                elif toks[0] == "1":
                    req_patterns["1 0 0 0 0 <13 tiles>  (deal; four flower counts always 0)"] += 1
                elif toks[0] == "2":
                    req_patterns["2 <tile>  (own draw)"] += 1
                else:
                    key = "3 <p> " + toks[2] + {
                        "DRAW": "", "PLAY": " <tile>",
                        "PENG": " <follow-up discard>",
                        "CHI": " <mid> <follow-up discard>",
                        "GANG": " <tile>" if len(toks) > 3 else "",
                        "BUGANG": " <tile>"}.get(toks[2], " ...")
                    req_patterns[key] += 1
        elif out.get("command") == "finish":
            fin_scores = [int(out["content"][str(s)]) for s in range(4)]
        # responses = the bot entry that follows this judge entry
        if i + 1 < len(logs) and isinstance(logs[i + 1], dict) \
                and "output" not in logs[i + 1]:
            resp = {}
            verdict_bad = []
            for k, v in logs[i + 1].items():
                if k in ("0", "1", "2", "3") and isinstance(v, dict):
                    resp[k] = v.get("response", "")
                    if v.get("verdict") != "OK":
                        verdict_bad.append((k, v.get("verdict")))
            if verdict_bad:
                raise CheckFail("non-OK verdict %s" % verdict_bad)
            turn["responses"] = resp
        turns.append(turn)

    displays = [t["display"] for t in turns]
    cls = classify_terminal(displays)
    last = displays[-1]
    if cls["ending"] == "draw":
        fan, fan_total = [], 0
        scores = fin_scores if fin_scores is not None else [0, 0, 0, 0]
    else:
        fan = [[f["name"], f["value"], f["cnt"]] for f in last["fan"]]
        fan_total = last["fanCnt"]
        scores = fin_scores if fin_scores is not None else list(last["score"])
    rec = {
        "game_id": raw["_mid"],
        "srand": init["srand"],
        "quan": init.get("quan", 0),
        "players": players,
        "wall": init["walltiles"].split(),
        "turns": turns,
        "expected": {
            "ending": cls["ending"],
            "winner": cls["winner"],
            "discarder": cls["discarder"],
            "qianggang": cls["qianggang"],
            "fan": fan,
            "fan_total": fan_total,
            "scores": scores,
        },
    }
    return rec, req_patterns


FAN_TAGS = {
    "杠上开花": "fan_gangshangkaihua",   # win on the kong replacement tile
    "抢杠和": "fan_qiangganghu",         # robbing the kong
    "妙手回春": "fan_last_tile_zimo",     # last wall tile, self-drawn
    "海底捞月": "fan_last_tile_ron",      # win on the last discard
}


def process_line(line):
    raw = json.loads(line)
    try:
        rec, req_patterns = build_record(raw)
        facts = check_record(rec)          # reference structural validation
    except CheckFail as e:
        return dict(game_id=raw.get("_mid"), error=str(e))
    except Exception as e:  # noqa: BLE001
        return dict(game_id=raw.get("_mid"), error="internal: %r" % e)

    exp = rec["expected"]
    tags = [exp["ending"]]
    if exp["qianggang"]:
        tags.append("qianggang")
    for key, tag in (("angang", "angang"), ("minggang", "minggang"),
                     ("bugang", "bugang"), ("multi_claim", "multi_claim"),
                     ("multi_hu", "multi_hu")):
        if facts.get(key):
            tags.append(tag)
    if exp["ending"] != "draw":
        if exp["fan_total"] == 8:
            tags.append("fan8_boundary")
        if exp["fan_total"] >= 48:
            tags.append("big_fan")
        for name, _, _ in exp["fan"]:
            if name in FAN_TAGS:
                tags.append(FAN_TAGS[name])

    # judge vs python-MahjongGB rescoring
    gb = dict(status="n/a")
    if exp["ending"] != "draw":
        try:
            calc = gb_probe([t["display"] for t in rec["turns"]], rec["quan"])
        except Exception as e:  # noqa: BLE001
            calc = ("probe_error", repr(e)[:200], None)
        if calc is None:
            gb = dict(status="probe_missing")
        elif calc[0] == "ok":
            fc_cai, fl_cai, fc_fix, fl_fix = calc[1], calc[2], calc[3], calc[4]
            if fc_cai == exp["fan_total"]:
                gb = dict(status="agree", fan=fc_cai)
            elif fc_fix == exp["fan_total"]:
                # MahjongGB agrees with the judge once is4thTile is computed
                # correctly; the cai-style FeatureAgent replay disagrees.
                st = ("cai_replay_below_8fan_gate" if fc_cai < 8
                      else "cai_replay_fan_diff")
                gb = dict(status=st, fan_cai_replay=fc_cai, fans_cai_replay=fl_cai,
                          fan_corrected=fc_fix)
                tags.append("gb_discrepancy" if fc_cai < 8 else "gb_fan_diff")
            else:
                gb = dict(status="unresolved_fan_diff", fan_cai_replay=fc_cai,
                          fans_cai_replay=fl_cai, fan_corrected=fc_fix,
                          fans_corrected=fl_fix)
                tags.append("gb_discrepancy" if fc_cai < 8 else "gb_fan_diff")
        else:
            gb = dict(status="calc_error", detail=calc[1])
            tags.append("gb_discrepancy")

    rec["tags"] = sorted(set(tags))
    if "gb_discrepancy" in rec["tags"] or "gb_fan_diff" in rec["tags"]:
        rec["known_discrepancy"] = {
            "note": "The official Botzone judge scored this win fan_total=%d. "
                    "A python-MahjongGB replay using the widely-copied cai-style "
                    "FeatureAgent flags scores it differently (below): that agent "
                    "computes is4thTile as shownTiles[winTile]==4, which can never "
                    "fire on a self-drawn win, so it misses 和绝张 (4 fan). With "
                    "is4thTile corrected, MahjongGB matches the judge "
                    "(fan_corrected). The judge is the ground truth for engine "
                    "acceptance; this record is included deliberately as a scorer "
                    "edge case." % exp["fan_total"],
            "mahjonggb": gb,
        }
    return dict(game_id=rec["game_id"], line=json.dumps(rec, ensure_ascii=False),
                tags=rec["tags"], fan_total=exp["fan_total"], gb=gb,
                req_patterns=dict(req_patterns), quan=rec["quan"])


GOLDEN_QUOTA = [
    ("qianggang", None), ("multi_hu", 20), ("gb_discrepancy", None),
    ("gb_fan_diff", 20),
    ("zimo", 20), ("ron", 20), ("draw", 20),
    ("angang", 15), ("minggang", 15), ("bugang", 15),
    ("multi_claim", 20), ("fan8_boundary", 15), ("big_fan", 10),
    ("fan_gangshangkaihua", 8), ("fan_qiangganghu", None),
    ("fan_last_tile_zimo", 5), ("fan_last_tile_ron", 5),
]


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(BASE + "/final2_all.jsonl.gz", "rt") as f:
        lines = f.readlines()
    print("loaded %d raw games (%.0fs)" % (len(lines), time.time() - t0), flush=True)

    with mp.Pool(min(100, mp.cpu_count())) as pool:
        results = pool.map(process_line, lines, chunksize=8)

    errors = [r for r in results if "error" in r]
    good = [r for r in results if "error" not in r]
    print("built %d records, %d errors (%.0fs)"
          % (len(good), len(errors), time.time() - t0), flush=True)
    for e in errors[:10]:
        print("  ERROR", e["game_id"], e["error"])

    # aggregate
    tag_counts = Counter()
    gb_counts = Counter()
    req_patterns = Counter()
    quan_counts = Counter()
    for r in good:
        tag_counts.update(r["tags"])
        gb_counts[r["gb"]["status"]] += 1
        req_patterns.update(r["req_patterns"])
        quan_counts[r["quan"]] += 1

    # golden selection (deterministic: sorted by game_id within each tag)
    by_tag = defaultdict(list)
    for r in good:
        for t in r["tags"]:
            by_tag[t].append(r)
    golden_ids = {}
    for tag, quota in GOLDEN_QUOTA:
        cands = sorted(by_tag.get(tag, []), key=lambda r: r["game_id"])
        if tag == "big_fan":
            cands = sorted(by_tag.get(tag, []),
                           key=lambda r: (-r["fan_total"], r["game_id"]))
        take = cands if quota is None else cands[:quota]
        for r in take:
            golden_ids.setdefault(r["game_id"], r)
    golden = sorted(golden_ids.values(), key=lambda r: r["game_id"])
    golden_tag_counts = Counter()
    for r in golden:
        golden_tag_counts.update(r["tags"])

    with gzip.open(OUT + "/mcr_final2026_full.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in good:
            f.write(r["line"] + "\n")
    with open(OUT + "/mcr_final2026_golden.jsonl", "w", encoding="utf-8") as f:
        for r in golden:
            f.write(r["line"] + "\n")

    disc = [dict(game_id=r["game_id"], fan_total=r["fan_total"], gb=r["gb"])
            for r in good if r["gb"]["status"] not in ("agree", "n/a")]
    summary = dict(
        n_games=len(good), n_build_errors=len(errors),
        build_errors=[dict(game_id=e["game_id"], error=e["error"]) for e in errors],
        tag_counts_full=dict(tag_counts.most_common()),
        n_golden=len(golden),
        tag_counts_golden=dict(golden_tag_counts.most_common()),
        golden_game_ids=[r["game_id"] for r in golden],
        gb_scan=dict(gb_counts),
        gb_discrepancies=disc,
        quan_distribution=dict(quan_counts),
        request_grammar=dict(req_patterns.most_common()),
        seconds=round(time.time() - t0, 1),
    )
    with open(OUT + "/TAGS_SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("golden_game_ids", "gb_discrepancies",
                                   "build_errors")},
                     indent=2, ensure_ascii=False), flush=True)
    print("gb_discrepancies: %d (see TAGS_SUMMARY.json)" % len(disc))
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
