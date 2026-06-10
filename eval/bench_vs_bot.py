"""
bench_vs_bot.py — head-to-head between two ARBITRARY Botzone bot commands through the
official judge (2xA + 2xB, seats rotated to cancel positional bias). Takes raw commands, so it
can pit our bot against a different-architecture bot (e.g. a top-30 imitation).

  python3 eval/bench_vs_bot.py "<cmdA>" "<cmdB>" [N] [nameA] [nameB]

PERSISTENT BOTS (2026-06-10): the 4 bot processes are built ONCE per matchup and reused across all
N games (the deploy bot re-inits its agent on every INIT request). This kills the per-game
57MB-model reload that caused the 7-20/72 stuck-rate noise — the noise that left the
lad_chunjiandu-vs-distill100b verdict inside the error bars. Dead/wedged bots are respawned
between games. Env: BENCH_TIMEOUT (s/turn, default 10), WALL_SEED_BASE (duplicate walls — use the
SAME base across candidates so they face identical deals), BENCH_DEBUG=1 (bot stderr to /tmp).
"""
import sys, os
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from run_match_kr import run_match_kr, make_bot
from data.log_collector import make_wall

A, B = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
nameA = sys.argv[4] if len(sys.argv) > 4 else "A"
nameB = sys.argv[5] if len(sys.argv) > 5 else "B"
TIMEOUT = float(os.environ.get("BENCH_TIMEOUT", "10"))

def spec(cmd): return {"cmd": cmd, "kr": True}

# 4 persistent processes: 2 per side, reused across all games (respawned only if dead).
procs = {"A1": None, "A2": None, "B1": None, "B2": None}
def ensure(k):
    if procs[k] is None or getattr(procs[k], "dead", False):
        if procs[k] is not None:
            procs[k].close()
        side = A if k[0] == "A" else B
        procs[k] = make_bot(spec(side), TIMEOUT, label=f"{nameA if k[0]=='A' else nameB}.{k}")
    return procs[k]

sA = sB = wA = wB = draws = illA = illB = stuck = played = 0
try:
    for g in range(N):
        if g % 2 == 0:   # A at seats 0,2
            aseats, order = [0, 2], ["A1", "B1", "A2", "B2"]
        else:            # A at seats 1,3
            aseats, order = [1, 3], ["B1", "A1", "B2", "A2"]
        bseats = [i for i in range(4) if i not in aseats]
        bots = [ensure(k) for k in order]
        specs = [spec(A if k[0] == "A" else B) for k in order]
        labels = [f"{nameA if k[0]=='A' else nameB}.{k}" for k in order]
        r = run_match_kr(specs, wall_json=make_wall(int(os.environ.get("WALL_SEED_BASE", "40000")) + g),
                         quan=0, timeout=TIMEOUT, labels=labels, bots=bots)
        if r.get("stuck"):
            stuck += 1
            continue     # dead bots respawn via ensure() next game
        played += 1
        sc = r["scores"]
        for s in aseats: sA += sc[s]
        for s in bseats: sB += sc[s]
        for s in range(4):
            if sc[s] == -30 and sum(1 for x in sc if x == 10) == 3:
                if s in aseats: illA += 1
                else: illB += 1
        w = max(range(4), key=lambda i: sc[i]) if max(sc) > 0 else -1
        if w == -1: draws += 1
        elif w in aseats: wA += 1
        else: wB += 1
        if (g + 1) % 10 == 0:
            print(f"  [{g+1}/{N}] {nameA} net={sA:+d} w={wA} | {nameB} net={sB:+d} w={wB} | draws={draws} stuck={stuck}", flush=True)
finally:
    for b in procs.values():
        if b is not None:
            b.close()

print(f"\n{N} games requested, {played} played, {stuck} stuck/skipped (2v2 rotated, official judge, persistent bots)")
print(f"  {nameA}: net={sA:+d}  wins={wA}  illegal={illA}")
print(f"  {nameB}: net={sB:+d}  wins={wB}  illegal={illB}")
print(f"  draws={draws}")
if played:
    print(f"  => {nameA if sA>sB else nameB if sB>sA else 'TIE'} stronger (net diff {sA-sB:+d})")
else:
    print("  => NO VALID GAMES (all stuck) — run with BENCH_DEBUG=1 and check /tmp/bench_err_*")
