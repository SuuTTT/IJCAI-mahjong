"""
replay_audit.py — P1 correctness suite over REAL histories, exercising the bot's full-history replay
path (run_json/_replay_event — Botzone's path, where the WH bug lived; the local keep-running eval
never tests it). Reuses the VALIDATED extraction (output.content[seat] + per-seat response records)
that reproduced the 5 sim-7 WH bugs.

Oracle (no fan-calc, no hand-tracking -> zero false positives): feed the bot (R[:k+1], RESP[:k]); if
the bot returns HU while the seat's real response RESP[k] was NOT HU, the bot phantom-wins where a
correct bot didn't = the -30 bug class. Also flags CRASH.

Modes:
  --mode preempt (default): only audit decisions at/after a PREEMPTED claim (recorded CHI/PENG/GANG
                            whose next request isn't the seat's confirming echo) — the exact trigger,
                            cheap enough to scan ALL games.
  --mode full  --limit N  : audit every decision point over N games (broad, O(L^2), sample it).

  python3 eval/replay_audit.py --root <dir> [--mode preempt|full] [--limit N] [--botdir ...]
"""
import os, sys, json, glob, argparse, io, contextlib

def load_bot(botdir):
    os.environ['MODEL'] = ''; os.environ['BOTZONE_JSON'] = '1'
    sys.path.insert(0, botdir)
    import importlib.util
    spec = importlib.util.spec_from_file_location('botmain', os.path.join(botdir, '__main__.py'))
    bot = importlib.util.module_from_spec(spec); sys.argv = ['x']
    with contextlib.redirect_stdout(io.StringIO()):
        try: spec.loader.exec_module(bot)
        except SystemExit: pass
    cap = {}
    bot.emit = lambda r: cap.__setitem__('r', r)
    def decide(R, RESP):
        cap['r'] = 'PASS'
        try: bot.run_json(json.dumps({'requests': R, 'responses': RESP}))
        except Exception as e: return 'CRASH:' + str(e)[:80]
        return cap.get('r', 'PASS') or 'PASS'
    return decide

def streams(d):
    """per-seat (R, RESP) from a full_log, like test_wh_fix (validated)."""
    out = [([], []) for _ in range(4)]
    for rec in d:
        if 'output' in rec:
            c = (rec['output'].get('content') or {})
            for s in range(4):
                if str(s) in c: out[s][0].append(str(c[str(s)]))
        elif isinstance(rec.get('0'), dict):
            for s in range(4):
                r = rec.get(str(s))
                if isinstance(r, dict) and len(out[s][1]) < len(out[s][0]):
                    out[s][1].append((r.get('response') or 'PASS').strip())
    return out

def real_winners(d):
    """seats that LEGITIMATELY won (final score > 0). A bot HU by any other seat is phantom."""
    win = set()
    for rec in d:
        disp = (rec.get('output') or {}).get('display') or {}
        sc = disp.get('score')
        if sc:
            win = {i for i in range(4) if sc[i] > 0}
    return win

def preempt_points(R, RESP):
    pts = []
    for k in range(min(len(R), len(RESP))):
        rp = RESP[k].split()
        if rp and rp[0] in ('CHI', 'PENG', 'GANG') and R[k].split()[:1] == ['3']:
            nt = R[k + 1].split() if k + 1 < len(R) else []
            if not (len(nt) >= 3 and nt[0] == '3' and nt[2] == rp[0]):
                pts += [k + 1, k + 2]      # the decision(s) right after the preempted claim
    return [k for k in dict.fromkeys(pts) if 0 < k < min(len(R), len(RESP))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--botdir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'deploy', 'caiest_cnn'))
    ap.add_argument('--mode', default='preempt', choices=['preempt', 'full'])
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    decide = load_bot(a.botdir)
    logs = sorted(glob.glob(os.path.join(a.root, '**', '*_full_log.json'), recursive=True))
    if a.limit: logs = logs[:a.limit]
    games = decpts = phantom = crash = preempt_games = 0; flags = []
    for gi, path in enumerate(logs):
        if os.path.getsize(path) == 0: continue
        try: d = json.load(open(path))
        except Exception: continue
        games += 1
        winners = real_winners(d)
        allstreams = streams(d)
        for s in range(4):
            R, RESP = allstreams[s]
            n = min(len(R), len(RESP))
            if n < 2: continue
            if a.mode == 'preempt':
                pts = preempt_points(R, RESP)
                if pts: preempt_games += 1
            else:
                pts = range(1, n)
            for k in pts:
                out = decide(R[:k + 1], RESP[:k]); decpts += 1
                w = out.split()[0]
                if out.startswith('CRASH:'):
                    flags.append(('CRASH', path, 's%d k%d %s' % (s, k, out))); crash += 1
                elif w == 'HU' and s not in winners:      # bot declares a win the seat never legitimately got
                    flags.append(('PHANTOM_HU', path, 's%d k%d real=%s' % (s, k, RESP[k]))); phantom += 1
        if games % 4000 == 0:
            print('  %d games, %d decpts, phantom=%d crash=%d' % (games, decpts, phantom, crash), flush=True)
    print('=== replay audit (%s): %d games, %d decision points, %d had preempted claims ===' % (a.mode, games, decpts, preempt_games))
    print('PHANTOM_HU: %d | CRASH: %d' % (phantom, crash))
    for f in flags[:25]: print('  ', f[0], os.path.basename(f[1])[:26], f[2])
    sys.exit(1 if (phantom or crash) else 0)

if __name__ == '__main__':
    main()
