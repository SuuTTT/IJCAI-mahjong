#!/usr/bin/env python3
"""Final2 diagnostic analysis over harvested match logs."""
import json, gzip, glob, os, math
from collections import defaultdict, Counter

BASE = "/root/final2_harvest"
BOTS = ["kong", "moyu", "QiuQiuR", "player152"]

def disp_stream(logs):
    for rec in logs:
        if not isinstance(rec, dict) or "output" not in rec:
            continue
        out = rec.get("output") or {}
        disp = out.get("display")
        if isinstance(disp, str):
            try:
                disp = json.loads(disp)
            except Exception:
                continue
        if isinstance(disp, dict) and disp.get("action"):
            yield rec, disp

games = []  # per-game summary dicts
verdict_counter = defaultdict(Counter)   # bot -> verdict counts (response records)
time_max = defaultdict(int)
time_sum = defaultdict(int); time_n = defaultdict(int)

files = sorted(glob.glob(f"{BASE}/raw/batch_*.jsonl.gz"))
seen = set()
for fp in files:
    with gzip.open(fp, "rt") as f:
        for line in f:
            d = json.loads(line)
            mid = d["_mid"]
            if mid in seen:
                continue
            seen.add(mid)
            logs = d.get("logs") or []
            # seat -> user
            users = []
            for p in d.get("players", []):
                nm = p.get("name", "")
                users.append(nm.split("]")[0].lstrip("[") if "]" in nm else nm)
            try:
                init = json.loads(d.get("initdata") or "{}")
            except Exception:
                init = {}
            srand = init.get("srand")
            quan = init.get("quan")

            # verdicts of bot responses + times
            err_seats = {}
            for rec in logs:
                if isinstance(rec, dict) and "output" not in rec:
                    for k, v in rec.items():
                        if k in "0123" and isinstance(v, dict):
                            s = int(k)
                            u = users[s] if s < len(users) else str(s)
                            vd = v.get("verdict", "?")
                            verdict_counter[u][vd] += 1
                            t = v.get("time")
                            if isinstance(t, (int, float)):
                                time_sum[u] += t; time_n[u] += 1
                                if t > time_max[u]:
                                    time_max[u] = t
                            if vd not in ("OK",):
                                err_seats.setdefault(s, vd)

            # walk displays for end info
            end_kind = "unknown"; winner = None; fan_cnt = None; fans = None
            scores = None; dealin_seat = None; zimo = False
            prev_disp = None; n_disc = 0
            last_disp = None
            for rec, disp in disp_stream(logs):
                a = disp["action"]
                if a == "PLAY":
                    n_disc += 1
                if a == "HU":
                    end_kind = "hu"
                    winner = disp.get("player")
                    fan_cnt = disp.get("fanCnt")
                    fans = disp.get("fan")
                    scores = disp.get("score")
                    if prev_disp is not None:
                        pa = prev_disp.get("action")
                        if pa == "DRAW" and prev_disp.get("player") == winner:
                            zimo = True
                        elif pa in ("PLAY", "BUGANG", "GANG", "PENG", "CHI"):
                            dealin_seat = prev_disp.get("player")
                    if scores is None and prev_disp is not None:
                        pass
                elif a == "HUANG":
                    end_kind = "huang"
                    scores = disp.get("score") or [0, 0, 0, 0]
                if a not in ("HU",):
                    prev_disp = disp
                last_disp = disp

            # finish record content as authoritative scores
            fin_scores = None
            for rec in reversed(logs):
                if isinstance(rec, dict) and "output" in rec:
                    out = rec["output"] or {}
                    if out.get("command") == "finish":
                        c = out.get("content") or {}
                        try:
                            fin_scores = [int(c[str(i)]) for i in range(4)]
                        except Exception:
                            fin_scores = None
                        break
            if fin_scores is not None:
                scores = fin_scores
            if end_kind == "unknown":
                end_kind = "error" if err_seats else "unknown"
            if err_seats:
                # error games: end may still be hu/huang if recoverable; flag separately
                pass

            games.append(dict(
                mid=mid, srand=srand, quan=quan, users=users,
                end=end_kind, winner=winner, zimo=zimo, dealin=dealin_seat,
                fan_cnt=fan_cnt,
                fans=[(f.get("name"), f.get("value"), f.get("cnt")) for f in fans] if fans else None,
                scores=scores, n_disc=n_disc,
                errs={str(k): v for k, v in err_seats.items()},
            ))

print(f"parsed {len(games)} games")
json_out = {"n_games": len(games)}

# save per-game summary
with gzip.open(f"{BASE}/final2_games_summary.jsonl.gz", "wt") as f:
    for g in games:
        f.write(json.dumps(g, ensure_ascii=False) + "\n")

# ---------- per-bot aggregates ----------
agg = {u: defaultdict(float) for u in BOTS}
fan_on_win = defaultdict(list)
dealin_cost = defaultdict(list)
scores_by = defaultdict(list)
seat_scores = {u: defaultdict(list) for u in BOTS}
placements = defaultdict(list)
comp = {u: defaultdict(float) for u in BOTS}   # score components
big_losses = []
end_counter = Counter()
err_games_by = Counter()
fan_names = defaultdict(Counter)

for g in games:
    if not g["scores"] or len(g["users"]) != 4:
        continue
    end_counter[g["end"]] += 1
    us = g["users"]; sc = g["scores"]
    ranked = sorted(range(4), key=lambda i: -sc[i])
    i = 0
    while i < 4:
        j = i
        while j < 4 and sc[ranked[j]] == sc[ranked[i]]:
            j += 1
        for k in range(i, j):
            placements[us[ranked[k]]].append((i + 1 + j) / 2)
        i = j
    for s in range(4):
        u = us[s]
        agg[u]["games"] += 1
        agg[u]["total"] += sc[s]
        scores_by[u].append(sc[s])
        seat_scores[u][s].append(sc[s])
        if sc[s] <= -60:
            big_losses.append((sc[s], u, g["mid"], g["end"], g["fan_cnt"], g["errs"]))
    for sstr, vd in g["errs"].items():
        u = us[int(sstr)]
        err_games_by[u] += 1
        agg[u]["err_games"] += 1
    if g["end"] == "hu" and g["winner"] is not None:
        w = g["winner"]; u = us[w]
        agg[u]["wins"] += 1
        if g["zimo"]:
            agg[u]["zimo"] += 1
        if g["fan_cnt"] is not None:
            fan_on_win[u].append(g["fan_cnt"])
        if g["fans"]:
            for nm, val, cnt in g["fans"]:
                fan_names[u][nm] += cnt or 1
        comp[u]["win_pts"] += sc[w]
        if g["dealin"] is not None and not g["zimo"]:
            dl = g["dealin"]; du = us[dl]
            agg[du]["dealins"] += 1
            dealin_cost[du].append(sc[dl])
            comp[du]["dealin_pay"] += sc[dl]
            for s in range(4):
                if s != w and s != dl:
                    comp[us[s]]["bystander_pay"] += sc[s]
        elif g["zimo"]:
            for s in range(4):
                if s != w:
                    comp[us[s]]["zimo_pay"] += sc[s]
        else:
            # hu without identified dealin (e.g. qiangang); treat payers
            for s in range(4):
                if s != w:
                    comp[us[s]]["other_pay"] += sc[s]
    elif g["end"] == "huang":
        for s in range(4):
            comp[us[s]]["draw_pts"] += sc[s]   # usually 0
    else:
        for s in range(4):
            comp[us[s]]["error_pts"] += sc[s]

per_bot = {}
for u in BOTS:
    n = agg[u]["games"]
    mean_t = (time_sum[u] / time_n[u]) if time_n[u] else None
    per_bot[u] = dict(
        games=int(n),
        total_score=int(agg[u]["total"]),
        mean_score=round(agg[u]["total"] / n, 4),
        mean_placement=round(sum(placements[u]) / len(placements[u]), 4),
        win_rate=round(agg[u]["wins"] / n, 4),
        zimo_rate=round(agg[u]["zimo"] / n, 4),
        ron_rate=round((agg[u]["wins"] - agg[u]["zimo"]) / n, 4),
        dealin_rate=round(agg[u]["dealins"] / n, 4),
        mean_fan_on_win=round(sum(fan_on_win[u]) / max(1, len(fan_on_win[u])), 2),
        mean_dealin_cost=round(sum(dealin_cost[u]) / max(1, len(dealin_cost[u])), 2),
        err_games=int(err_games_by[u]),
        err_game_rate=round(err_games_by[u] / n, 5),
        verdicts={k: v for k, v in verdict_counter[u].items() if k != "OK"},
        mean_resp_ms=round(mean_t, 1) if mean_t else None,
        max_resp_ms=int(time_max[u]),
        score_var=round(sum((x - agg[u]["total"] / n) ** 2 for x in scores_by[u]) / n, 2),
        seat_mean={s: round(sum(v) / len(v), 3) for s, v in sorted(seat_scores[u].items())},
        components_per_game={k: round(v / n, 4) for k, v in comp[u].items()},
        top_fans=fan_names[u].most_common(12),
    )

json_out["end_kinds"] = dict(end_counter)
json_out["per_bot"] = per_bot

# ---------- duplicate-wall paired analysis ----------
bywall = defaultdict(list)
for g in games:
    if g["srand"] is not None and g["scores"] and len(g["users"]) == 4:
        bywall[g["srand"]].append(g)
json_out["n_walls"] = len(bywall)
json_out["perm_sizes"] = dict(Counter(len(v) for v in bywall.values()))

# cell = (wall, seat) -> bot -> list of (score, outcome-tags)
pair_stats = {}
for a in BOTS:
    for b in BOTS:
        if a >= b:
            continue
        pair_stats[(a, b)] = dict(cells=0, diff_sum=0.0, a_better=0, b_better=0, tie=0)

cell_records = []  # for decomposition moyu vs kong
for wall, gs in bywall.items():
    cell = defaultdict(lambda: defaultdict(list))  # seat -> bot -> [game dicts]
    for g in gs:
        for s in range(4):
            cell[s][g["users"][s]].append(g)
    for s in range(4):
        for (a, b), st in pair_stats.items():
            if a in cell[s] and b in cell[s]:
                ma = sum(x["scores"][s] for x in cell[s][a]) / len(cell[s][a])
                mb = sum(x["scores"][s] for x in cell[s][b]) / len(cell[s][b])
                st["cells"] += 1
                st["diff_sum"] += ma - mb
                if ma > mb: st["a_better"] += 1
                elif mb > ma: st["b_better"] += 1
                else: st["tie"] += 1
        # record for kong vs moyu decomposition
        if "kong" in cell[s] and "moyu" in cell[s]:
            def summarize(glist, seat):
                r = dict(n=len(glist), score=0.0, wins=0, fan=0.0, dealins=0,
                         dealin_pay=0.0, win_pts=0.0, errs=0, other_pay=0.0)
                for x in glist:
                    sc = x["scores"][seat]
                    r["score"] += sc
                    if x["end"] == "hu" and x["winner"] == seat:
                        r["wins"] += 1; r["fan"] += (x["fan_cnt"] or 0); r["win_pts"] += sc
                    elif x["end"] == "hu" and x["dealin"] == seat and not x["zimo"]:
                        r["dealins"] += 1; r["dealin_pay"] += sc
                    else:
                        r["other_pay"] += sc
                    if str(seat) in x["errs"]:
                        r["errs"] += 1
                for k in ("score", "fan", "dealin_pay", "win_pts", "other_pay"):
                    r[k] /= r["n"]
                for k in ("wins", "dealins", "errs"):
                    r[k] = r[k] / r["n"]
                return r
            cell_records.append(dict(
                wall=wall, seat=s,
                kong=summarize(cell[s]["kong"], s),
                moyu=summarize(cell[s]["moyu"], s),
            ))

pair_out = {}
for (a, b), st in pair_stats.items():
    pair_out[f"{a}_vs_{b}"] = dict(
        cells=st["cells"],
        mean_paired_diff=round(st["diff_sum"] / max(1, st["cells"]), 4),
        a_better=st["a_better"], b_better=st["b_better"], tie=st["tie"],
    )
json_out["paired_walls"] = pair_out

# decomposition kong - moyu over matched cells
dec = defaultdict(float); N = len(cell_records)
for c in cell_records:
    k, m = c["kong"], c["moyu"]
    dec["score"] += k["score"] - m["score"]
    dec["win_rate"] += k["wins"] - m["wins"]
    dec["win_pts"] += k["win_pts"] - m["win_pts"]
    dec["dealin_rate"] += k["dealins"] - m["dealins"]
    dec["dealin_pay"] += k["dealin_pay"] - m["dealin_pay"]
    dec["other_pay"] += k["other_pay"] - m["other_pay"]
    dec["err_rate"] += k["errs"] - m["errs"]
json_out["kong_minus_moyu_paired"] = {k: round(v / max(1, N), 5) for k, v in dec.items()}
json_out["kong_minus_moyu_cells"] = N

# head-to-head: P(row finishes strictly above col)
h2h = {}
above = defaultdict(int); tot2 = defaultdict(int)
for g in games:
    if not g["scores"] or len(g["users"]) != 4:
        continue
    for i in range(4):
        for j in range(4):
            if i == j: continue
            a, b = g["users"][i], g["users"][j]
            tot2[(a, b)] += 1
            if g["scores"][i] > g["scores"][j]:
                above[(a, b)] += 1
for a in BOTS:
    h2h[a] = {b: (round(above[(a, b)] / tot2[(a, b)], 4) if tot2[(a, b)] else None)
              for b in BOTS if b != a}
json_out["head_to_head_above_rate"] = h2h

big_losses.sort()
json_out["biggest_losses"] = [
    dict(score=x[0], bot=x[1], mid=x[2], end=x[3], fan=x[4], errs=x[5])
    for x in big_losses[:20]
]

json.dump(json_out, open(f"{BASE}/FINAL2_DIAGNOSIS.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(json_out, indent=2, ensure_ascii=False)[:8000])
print("saved FINAL2_DIAGNOSIS.json")
