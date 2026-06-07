"""
test_wh_fix.py — regression test for the preempted-claim replay fix, driven by REAL
Botzone tournament logs.

For each game, rebuilds the per-turn JSON history (requests R[:k+1], responses RESP[:k])
exactly as Botzone feeds it, pipes every decision point through ONE persistent bot process
(JSON mode), and checks:
  - WH games (listed via --wh): the final decision must NOT be HU (the old code phantom-HUed)
  - clean games: every decision must reproduce the recorded response (no behavior change)

  python3 eval/test_wh_fix.py --botdir /root/mahjong/botFix --root games_sim7/moyu \
      --name "[moyu]caiest" --clean 40 --wh 6a2460ca5eab685a5f7c99dc,...
"""
import os, sys, json, glob, argparse, subprocess

ap = argparse.ArgumentParser()
ap.add_argument('--botdir', required=True)
ap.add_argument('--root', required=True)
ap.add_argument('--name', default='[moyu]caiest')
ap.add_argument('--clean', type=int, default=40)
ap.add_argument('--wh', default='')
a = ap.parse_args()
WH = set(x for x in a.wh.split(',') if x)

def game_stream(mid_glob):
    """yield (mid, requests, responses) for our seat."""
    for mp in sorted(glob.glob(os.path.join(a.root, '**', '*_metadata.json'), recursive=True)):
        mid = os.path.basename(mp).split('_')[0]
        if mid_glob and mid not in mid_glob: continue
        try: md = json.load(open(mp))
        except Exception: continue
        names = [p.get('name') for p in md.get('players', [])]
        if a.name not in names: continue
        seat = names.index(a.name)
        fl = [p for p in glob.glob(os.path.join(os.path.dirname(mp), mid + '_*full_log.json')) if os.path.getsize(p) > 0]
        if not fl: continue
        try: d = json.load(open(fl[0]))
        except Exception: continue
        R, RESP = [], []
        for rec in d:
            if 'output' in rec:
                c = (rec['output'].get('content') or {})
                if str(seat) in c: R.append(c[str(seat)])
            elif '0' in rec:
                r = rec.get(str(seat))
                if isinstance(r, dict) and len(RESP) < len(R):
                    RESP.append((r.get('response') or 'PASS').strip())
        yield mid, R, RESP

bot = subprocess.Popen(['python3', '__main__.py'], cwd=a.botdir,
                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                       env={**os.environ, 'BOTZONE_JSON': '1', 'OPENBLAS_NUM_THREADS': '1'})

def ask(R, RESP):
    bot.stdin.write(json.dumps({'requests': R, 'responses': RESP}) + '\n'); bot.stdin.flush()
    return json.loads(bot.stdout.readline())['response'].strip()

wh_pass = wh_fail = 0
for mid, R, RESP in game_stream(WH):
    out = ask(R, RESP[:len(R) - 1])     # the WH decision point (last request)
    ok = out != 'HU'
    wh_pass += ok; wh_fail += (not ok)
    print(f"WH {mid[-6:]}: old=HU new={out} {'PASS' if ok else 'FAIL'}")

def has_preempted_claim(R, RESP):
    """A recorded CHI/PENG/GANG on a discard whose next request is not our confirming echo."""
    for k in range(min(len(R), len(RESP))):
        rp = RESP[k].split()
        if rp and rp[0] in ('CHI', 'PENG', 'GANG') and R[k].split()[0] == '3':
            nt = R[k + 1].split() if k + 1 < len(R) else []
            if not (len(nt) >= 3 and nt[0] == '3' and nt[2] == rp[0]):
                return True
    return False

mism = ngames = ndec = npre = 0
for mid, R, RESP in game_stream(None):
    if mid in WH: continue
    if has_preempted_claim(R, RESP):
        npre += 1; continue        # recorded moves came from the (old) corrupted state — skip
    if ngames >= a.clean: continue
    ngames += 1
    for k in range(1, min(len(R), len(RESP))):
        out = ask(R[:k + 1], RESP[:k]); ndec += 1
        if out != RESP[k]:
            mism += 1
            if mism <= 5: print(f"  mismatch {mid[-6:]} turn {k}: recorded={RESP[k]!r} new={out!r}")
print(f"games with preempted claims (besides WH): {npre}")
bot.stdin.close(); bot.terminate()
print(f"\nWH games: {wh_pass} fixed, {wh_fail} still HU")
print(f"clean replay: {ngames} games, {ndec} decisions, {mism} mismatches")
sys.exit(0 if (wh_fail == 0 and mism == 0) else 1)
