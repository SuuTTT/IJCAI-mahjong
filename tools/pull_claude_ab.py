"""
pull_claude_ab.py — fetch [Claude]aaa (net-PIMC) games from Botzone, save to claude_games/, and
report the clean A/B vs [moyu]distill (plain) in the same tables, with a paired-diff significance
check. Crashes (our seat == -30 forfeit) are excluded. Run repeatedly; it accumulates.

  python3 tools/pull_claude_ab.py            # one pull + report
"""
import urllib.request, json, re, os, glob, math, datetime, sys

BASE = "https://www.botzone.org.cn"; GID = "5e37dcf74019f43051e53201"
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'others/ladder_top30_score1216/claude_games')
os.makedirs(D, exist_ok=True)
CUTOFF = int(sys.argv[1], 16) if len(sys.argv) > 1 else 0x6a2d3000   # don't scan older than this ts


def fetch(url):
    r = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"})
    return urllib.request.urlopen(r, timeout=60).read().decode('utf-8', 'ignore')


def pull(max_pages=40):
    have = set(os.path.basename(f).split('_')[0] for f in glob.glob(D + '/*_full_log.json'))
    url = f"{BASE}/globalmatchlist?game={GID}"; mids = []; pages = 0
    while url and pages < max_pages:
        html = fetch(url); pages += 1; stop = False
        for mid in dict.fromkeys(re.findall(r'/match/([0-9a-f]{24})', html)):
            if int(mid[:8], 16) < CUTOFF: stop = True; break
            mids.append(mid)
        if stop: break
        nx = re.search(r'/globalmatchlist\?startid=[0-9a-f]+[^"]*game=' + GID, html)
        url = BASE + nx.group(0) if nx else None
    got = 0
    for mid in dict.fromkeys(mids):
        if mid in have: continue
        try: d = json.loads(fetch(f"{BASE}/match/{mid}?lite=true"))
        except Exception: continue
        names = [p.get('name', '?') for p in d.get('players', [])]
        if not any('Claude]aaa' in n for n in names): continue
        open(f'{D}/{mid}_full_log.json', 'w').write(json.dumps(d.get('logs') or []))
        open(f'{D}/{mid}_metadata.json', 'w').write(json.dumps({"matchId": mid, "players": [{"name": n} for n in names]}))
        got += 1
    return got


def final(logs):
    sc = act = winner = lastplay = None
    for rec in logs:
        if not isinstance(rec, dict): continue
        disp = (rec.get('output') or {}).get('display') or {}
        if disp.get('action') == 'PLAY': lastplay = disp.get('player')
        if isinstance(disp.get('score'), list): sc, act, winner = disp['score'], disp.get('action'), disp.get('player')
    return sc, act, winner, lastplay


def report():
    ab = []; field = []
    for mp in glob.glob(D + '/*_metadata.json'):
        mid = os.path.basename(mp).split('_')[0]
        names = [p['name'] for p in json.load(open(mp))['players']]
        try: logs = json.load(open(f'{D}/{mid}_full_log.json'))
        except Exception: continue
        sc, act, winner, lastplay = final(logs)
        if not sc: continue
        ci = [i for i, n in enumerate(names) if 'Claude]aaa' in n][0]
        if sc[ci] == -30: continue                                   # crash forfeit
        mo = [i for i, n in enumerate(names) if 'moyu]distill' in n]
        if mo: ab.append((int(mid[:8], 16), sc[ci], sc[mo[0]]))
        else: field.append(sc[ci])
    ab.sort()
    out = ["=== %s ===" % datetime.datetime.utcnow().strftime('%m-%d %H:%M UTC')]
    if ab:
        diffs = [a - m for _, a, m in ab]
        na, nm = sum(a for _, a, _ in ab), sum(m for _, _, m in ab)
        mean = sum(diffs) / len(diffs)
        sd = (sum((x - mean) ** 2 for x in diffs) / max(len(diffs) - 1, 1)) ** 0.5
        se = sd / math.sqrt(len(diffs)) if len(diffs) > 1 else float('inf')
        z = mean / se if se else 0
        w = sum(1 for d in diffs if d > 0); t = sum(1 for d in diffs if d == 0)
        out.append("A/B (net-PIMC vs plain, same table): %d games" % len(ab))
        out.append("  net-PIMC %+d vs plain %+d | diff %+d | mean %+.1f/g (z=%.2f, %s)"
                   % (na, nm, na - nm, mean, z, 'SIGNIFICANT' if abs(z) > 2 else 'not yet sig'))
        out.append("  net-PIMC wins %d, ties %d, losses %d" % (w, t, len(ab) - w - t))
    else:
        out.append("no clean A/B games yet")
    if field:
        out.append("net-PIMC vs-field (no plain): %d games, net %+d, avg %+.1f/g"
                   % (len(field), sum(field), sum(field) / len(field)))
    return "\n".join(out)


if __name__ == '__main__':
    n = pull()
    print("pulled +%d new games" % n)
    print(report())
