"""
compare_models.py — relative strength test between two model checkpoints.
Seats 2x modelA + 2x modelB at a table over many fixed walls and compares total
score. Stronger policy accumulates more points across identical walls — a valid
relative measure even when many games draw (it only depends on the games where
the two policies actually interact differently).

Usage:
  OPENBLAS_NUM_THREADS=1 python3 eval/compare_models.py A.npz B.npz [N_games]
"""
import sys, os
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from run_match_kr import run_match_kr
from data.log_collector import make_wall

A = sys.argv[1]; B = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 40

def bot(m): return {"cmd": f"MODEL={m} python3 bot/ml_bot.py", "kr": True}

# Two seatings to cancel positional bias: A,B,A,B and B,A,B,A
layouts = [([0,2], [bot(A),bot(B),bot(A),bot(B)]),
           ([1,3], [bot(B),bot(A),bot(B),bot(A)])]

scoreA = scoreB = winA = winB = draws = illegalA = illegalB = 0
for g in range(N):
    aseats, bots = layouts[g % 2]
    bseats = [i for i in range(4) if i not in aseats]
    r = run_match_kr(bots, wall_json=make_wall(40000 + g), quan=0, timeout=8)
    sc = r["scores"]
    for s in aseats: scoreA += sc[s]
    for s in bseats: scoreB += sc[s]
    # illegal attribution
    for s in range(4):
        if sc[s] == -30 and sum(1 for x in sc if x == 10) == 3:
            (globals().__setitem__('illegalA', illegalA+1) if s in aseats
             else globals().__setitem__('illegalB', illegalB+1))
    w = max(range(4), key=lambda i: sc[i]) if max(sc) > 0 else -1
    if w == -1: draws += 1
    elif w in aseats: winA += 1
    else: winB += 1

nameA, nameB = os.path.basename(A), os.path.basename(B)
print(f"{N} games (2v2, seats rotated)")
print(f"  A={nameA}: net={scoreA:+d}  wins={winA}  illegal={illegalA}")
print(f"  B={nameB}: net={scoreB:+d}  wins={winB}  illegal={illegalB}")
print(f"  draws={draws}")
print(f"  => {'A stronger' if scoreA>scoreB else 'B stronger' if scoreB>scoreA else 'tie'} "
      f"(net diff {scoreA-scoreB:+d})")
