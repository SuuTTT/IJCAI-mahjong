#!/usr/bin/env python3
"""Wall-level significance + cell-category decomposition for Final2."""
import json, gzip, math, random
from collections import defaultdict

BASE = "/root/final2_harvest"
BOTS = ["kong", "moyu", "QiuQiuR", "player152"]

games = []
with gzip.open(f"{BASE}/final2_games_summary.jsonl.gz", "rt") as f:
    for line in f:
        games.append(json.loads(line))

bywall = defaultdict(list)
for g in games:
    bywall[g["srand"]].append(g)

# per-wall per-bot total score (each bot plays 24 games per wall, 6 per seat)
wall_tot = {}
for w, gs in bywall.items():
    t = defaultdict(float)
    for g in gs:
        for s in range(4):
            t[g["users"][s]] += g["scores"][s]
    wall_tot[w] = t

walls = list(wall_tot)
out = {}
random.seed(0)
for a in BOTS:
    for b in BOTS:
        if a >= b:
            continue
        diffs = [wall_tot[w][a] - wall_tot[w][b] for w in walls]
        n = len(diffs)
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        se = math.sqrt(var / n)
        t = mean / se if se else 0
        # bootstrap P(a total > b total)
        wins = 0
        B = 20000
        for _ in range(B):
            s = sum(diffs[random.randrange(n)] for _ in range(n))
            if s > 0:
                wins += 1
        out[f"{a}_vs_{b}"] = dict(
            mean_per_wall=round(mean, 3), se=round(se, 3), t=round(t, 3),
            total=round(mean * n, 1), boot_P_a_wins=round(wins / B, 4))

# ---- cell categories: kong vs moyu at same (wall, seat) ----
cat = defaultdict(float)
catn = defaultdict(int)
ncells = 0
fan_when_both_win = {"kong": [], "moyu": []}
for w, gs in bywall.items():
    cell = defaultdict(lambda: defaultdict(list))
    for g in gs:
        for s in range(4):
            cell[s][g["users"][s]].append(g)
    for s in range(4):
        if "kong" not in cell[s] or "moyu" not in cell[s]:
            continue
        ncells += 1
        def stats(botgames):
            wins = fans = winpts = dealins = dealpay = other = score = 0.0
            n = len(botgames)
            for x in botgames:
                sc = x["scores"][s]
                score += sc
                if x["end"] == "hu" and x["winner"] == s:
                    wins += 1; fans += (x["fan_cnt"] or 0); winpts += sc
                elif x["end"] == "hu" and x["dealin"] == s and not x["zimo"]:
                    dealins += 1; dealpay += sc
                else:
                    other += sc
            return dict(n=n, win=wins/n, fan=fans/max(1,wins), winpts=winpts/n,
                        dealin=dealins/n, dealpay=dealpay/n, other=other/n, score=score/n)
        k = stats(cell[s]["kong"]); m = stats(cell[s]["moyu"])
        cat["d_score"] += k["score"] - m["score"]
        cat["d_winrate"] += k["win"] - m["win"]
        cat["d_winpts"] += k["winpts"] - m["winpts"]
        cat["d_dealin_rate"] += k["dealin"] - m["dealin"]
        cat["d_dealin_pay"] += k["dealpay"] - m["dealpay"]
        cat["d_other_pay"] += k["other"] - m["other"]
        if k["win"] > 0 and m["win"] > 0:
            fan_when_both_win["kong"].append(k["fan"])
            fan_when_both_win["moyu"].append(m["fan"])

dec = {k: round(v / ncells, 5) for k, v in cat.items()}
nb = len(fan_when_both_win["kong"])
dec["cells"] = ncells
dec["fan_both_win_kong"] = round(sum(fan_when_both_win["kong"]) / nb, 3)
dec["fan_both_win_moyu"] = round(sum(fan_when_both_win["moyu"]) / nb, 3)
dec["cells_both_won"] = nb

# further split winpts gap into rate vs size:
# winpts = winrate * mean_win_size. delta approx.
res = dict(wall_level=out, kong_moyu_cells=dec)

# per-bot mean win size (points per win) overall
winsz = defaultdict(list)
for g in games:
    if g["end"] == "hu" and g["winner"] is not None:
        u = g["users"][g["winner"]]
        winsz[u].append(g["scores"][g["winner"]])
res["mean_win_size"] = {u: round(sum(v)/len(v), 3) for u, v in winsz.items()}
res["draw_rate"] = round(sum(1 for g in games if g["end"] == "huang") / len(games), 5)

# zimo vs ron win-size split
zsz = defaultdict(list); rsz = defaultdict(list)
for g in games:
    if g["end"] == "hu" and g["winner"] is not None:
        u = g["users"][g["winner"]]
        (zsz if g["zimo"] else rsz)[u].append(g["scores"][g["winner"]])
res["zimo_win_size"] = {u: round(sum(v)/len(v), 2) for u, v in zsz.items()}
res["ron_win_size"] = {u: round(sum(v)/len(v), 2) for u, v in rsz.items()}
res["zimo_n"] = {u: len(v) for u, v in zsz.items()}
res["ron_n"] = {u: len(v) for u, v in rsz.items()}

json.dump(res, open(f"{BASE}/final2_stats2.json", "w"), indent=2)
print(json.dumps(res, indent=2))
