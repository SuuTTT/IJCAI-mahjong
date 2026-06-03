"""
bench_vs_bot.py — head-to-head between two ARBITRARY Botzone bot commands through the
official judge (2xA + 2xB, seats rotated to cancel positional bias). Unlike compare_models
(which wraps both sides as ml_bot), this takes raw commands, so it can pit our r18 ml_bot
against a different-architecture bot (e.g. the reproduced caiest CNN).

  python3 eval/bench_vs_bot.py "<cmdA>" "<cmdB>" [N] [nameA] [nameB]
"""
import sys, os
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from run_match_kr import run_match_kr
from data.log_collector import make_wall

A, B = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
nameA = sys.argv[4] if len(sys.argv) > 4 else "A"
nameB = sys.argv[5] if len(sys.argv) > 5 else "B"

def bot(cmd): return {"cmd": cmd, "kr": True}
layouts = [([0, 2], [bot(A), bot(B), bot(A), bot(B)]),
           ([1, 3], [bot(B), bot(A), bot(B), bot(A)])]

sA = sB = wA = wB = draws = illA = illB = 0
for g in range(N):
    aseats, bots = layouts[g % 2]
    bseats = [i for i in range(4) if i not in aseats]
    r = run_match_kr(bots, wall_json=make_wall(40000 + g), quan=0, timeout=10)
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
        print(f"  [{g+1}/{N}] {nameA} net={sA:+d} w={wA} | {nameB} net={sB:+d} w={wB} | draws={draws}", flush=True)

print(f"\n{N} games (2v2 rotated, official judge)")
print(f"  {nameA}: net={sA:+d}  wins={wA}  illegal={illA}")
print(f"  {nameB}: net={sB:+d}  wins={wB}  illegal={illB}")
print(f"  draws={draws} ({100*draws/N:.0f}%)")
print(f"  => {nameA if sA>sB else nameB if sB>sA else 'TIE'} stronger (net diff {sA-sB:+d})")
