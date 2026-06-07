"""
diff_bots.py — feed identical per-turn JSON histories to TWO bot builds and diff their decisions.
Used to prove the WH fix is behavior-preserving on games WITHOUT preempted claims (where the
codepaths must be byte-identical), independent of any environment delta vs Botzone recordings.

  python3 eval/diff_bots.py --bota botOld --botb botFix --root games_sim7/moyu \
      --name "[moyu]caiest" --games 30 [--skip-preempted]
"""
import os, sys, json, glob, argparse, subprocess

ap = argparse.ArgumentParser()
ap.add_argument('--bota', required=True); ap.add_argument('--botb', required=True)
ap.add_argument('--root', required=True); ap.add_argument('--name', default='[moyu]caiest')
ap.add_argument('--games', type=int, default=30)
ap.add_argument('--skip-preempted', action='store_true')
a = ap.parse_args()

def spawn(d):
    return subprocess.Popen(['python3', '__main__.py'], cwd=d,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                            env={**os.environ, 'BOTZONE_JSON': '1', 'OPENBLAS_NUM_THREADS': '1'})

def ask(bot, R, RESP):
    bot.stdin.write(json.dumps({'requests': R, 'responses': RESP}) + '\n'); bot.stdin.flush()
    return json.loads(bot.stdout.readline())['response'].strip()

def preempted(R, RESP):
    for k in range(min(len(R), len(RESP))):
        rp = RESP[k].split()
        if rp and rp[0] in ('CHI', 'PENG', 'GANG') and R[k].split()[0] == '3':
            nt = R[k + 1].split() if k + 1 < len(R) else []
            if not (len(nt) >= 3 and nt[0] == '3' and nt[2] == rp[0]):
                return True
    return False

A, B = spawn(a.bota), spawn(a.botb)
ng = nd = ndiff = npre_diff = 0
for mp in sorted(glob.glob(os.path.join(a.root, '**', '*_metadata.json'), recursive=True)):
    if ng >= a.games: break
    mid = os.path.basename(mp).split('_')[0]
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
            if str(seat) in c: R.append(str(c[str(seat)]))
        elif '0' in rec:
            r = rec.get(str(seat))
            if isinstance(r, dict) and len(RESP) < len(R): RESP.append((r.get('response') or 'PASS').strip())
    pre = preempted(R, RESP)
    if a.skip_preempted and pre: continue
    ng += 1
    for k in range(1, min(len(R), len(RESP))):
        ra, rb = ask(A, R[:k + 1], RESP[:k]), ask(B, R[:k + 1], RESP[:k]); nd += 1
        if ra != rb:
            ndiff += 1; npre_diff += pre
            if ndiff <= 8: print(f"  diff {mid[-6:]}{' (PREEMPTED game)' if pre else ''} turn {k}: old={ra!r} new={rb!r}")
for b in (A, B): b.stdin.close(); b.terminate()
print(f"\n{ng} games, {nd} decisions: {ndiff} old-vs-new diffs ({npre_diff} of them in preempted games)")
sys.exit(0 if ndiff == npre_diff else 1)   # diffs allowed ONLY in preempted games
