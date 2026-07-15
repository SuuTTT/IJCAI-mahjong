import re, json, gzip
from collections import defaultdict

h = open("/root/final2_harvest/table.html", encoding="utf-8", errors="replace").read()
# Rows may be embedded in a JS string with escaped quotes/newlines.
h = h.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\/", "/")

rows = re.split(r"<tr>", h)
out = []
for r in rows:
    m = re.search(r"/match/([0-9a-f]{24})", r)
    if not m:
        continue
    mid = m.group(1)
    ts = re.search(r"<td>(\d{4}-\d+-\d+ \d+:\d+:\d+)</td>", r)
    scores = re.findall(r'<div class="score pull-right">(-?\d+)</div>', r)
    users = re.findall(r'<a class="smallusername"[^>]*href="/account/([0-9a-f]+)">([^<]*)</a>', r)
    bots = re.findall(r'<a class="botname[^"]*">([^<]*?)\s*<span', r)
    trailing = re.findall(r"<td>(\d+)</td>\s*</tr>", r)
    out.append({
        "mid": mid,
        "ts": ts.group(1) if ts else None,
        "scores": [int(s) for s in scores],
        "users": [u[1] for u in users],
        "uids": [u[0] for u in users],
        "bots": bots,
        "trailing": trailing[-1] if trailing else None,
    })

print("rows:", len(out))
bad = [o for o in out if len(o["scores"]) != 4 or len(o["users"]) != 4]
print("bad rows:", len(bad))
for b in bad[:2]:
    print(b)

with gzip.open("/root/final2_harvest/table_rows.jsonl.gz", "wt") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")

tot = defaultdict(int); n = defaultdict(int); plc = defaultdict(list)
wins = defaultdict(int)
for o in out:
    if len(o["scores"]) != 4:
        continue
    pairs = list(zip(o["users"], o["scores"]))
    ranked = sorted(pairs, key=lambda x: -x[1])
    i = 0
    rankmap = {}
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][1] == ranked[i][1]:
            j += 1
        avg = (i + 1 + j) / 2
        for k in range(i, j):
            rankmap[k] = avg
        i = j
    for idx, (u, s) in enumerate(ranked):
        plc[u].append(rankmap[idx])
    for u, s in pairs:
        tot[u] += s; n[u] += 1

res = {}
for u in tot:
    res[u] = {
        "games": n[u],
        "total_score": tot[u],
        "mean_score": round(tot[u] / n[u], 4),
        "mean_placement": round(sum(plc[u]) / len(plc[u]), 4),
        "first_places": sum(1 for p in plc[u] if p == 1),
    }
res = dict(sorted(res.items(), key=lambda kv: -kv[1]["total_score"]))
print(json.dumps(res, indent=2, ensure_ascii=False))
json.dump(res, open("/root/final2_harvest/final2_standings.json", "w"), indent=2, ensure_ascii=False)
