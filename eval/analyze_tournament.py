"""
analyze_tournament.py — audit OUR bot's games from Botzone tournament full logs.
  python3 eval/analyze_tournament.py --root <dir with *_full_log.json + *_metadata.json> --name "[moyu]caiest"
Reports: placement distribution, mean score, verdict counts for our seat (OK/TLE/RE/...),
max time/memory, wins (+fan), deal-ins (we discarded the winning tile), our wrong-HUs, draws.
"""
import os, json, glob, argparse, collections

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--name', default='[moyu]caiest')
a = ap.parse_args()

games = 0; ranks = collections.Counter(); verd = collections.Counter()
tmax = 0; mmax = 0; deal_in = 0; wins = 0; whs = 0; draws = 0
scores = []; fan_when_win = []; finish_actions = collections.Counter()
bad_verdict_games = []

for mp in sorted(glob.glob(os.path.join(a.root, '**', '*_metadata.json'), recursive=True)):
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
    games += 1
    last_play_p = None
    for rec in d:
        if not isinstance(rec, dict): continue
        if 'output' in rec:
            out = rec.get('output') or {}
            disp = out.get('display') or {}
            actn = disp.get('action')
            if actn == 'PLAY': last_play_p = disp.get('player')
            if out.get('command') == 'finish':
                finish_actions[actn or '?'] += 1
                sc = disp.get('score')
                if sc:
                    scores.append(sc[seat])
                    ranks[1 + sum(1 for i in range(4) if sc[i] > sc[seat])] += 1
                if actn == 'HU':
                    if disp.get('player') == seat:
                        wins += 1; fan_when_win.append(disp.get('fanCnt'))
                    elif last_play_p == seat:
                        deal_in += 1
                elif actn == 'WH' and disp.get('player') == seat:
                    whs += 1
                elif actn in ('HUANG', 'DRAW'):
                    draws += 1
        elif '0' in rec:  # players' responses keyed by seat
            r = rec.get(str(seat))
            if isinstance(r, dict):
                v = r.get('verdict')
                if v:
                    verd[v] += 1
                    if v not in ('OK',): bad_verdict_games.append((mid, v))
                tmax = max(tmax, r.get('time', 0) or 0)
                mmax = max(mmax, r.get('memory', 0) or 0)

n = max(1, games)
print(f"games {games} | mean score {sum(scores)/max(1,len(scores)):+.2f} | total {sum(scores)}")
print("rank dist:", {k: f"{v} ({100*v/n:.0f}%)" for k, v in sorted(ranks.items())})
print("verdicts (our seat):", dict(verd))
print(f"max time(ms) {tmax} | max memory(MB) {mmax}")
print(f"wins {wins} ({100*wins/n:.1f}%) fans={collections.Counter(fan_when_win)}")
print(f"deal-in {deal_in} ({100*deal_in/n:.1f}%) | our WH {whs} | draws {draws} ({100*draws/n:.1f}%)")
print("finish actions:", dict(finish_actions))
if bad_verdict_games[:10]:
    print("non-OK verdict games (first 10):", bad_verdict_games[:10])
