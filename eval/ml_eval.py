"""
ml_eval.py — fast ML-bot evaluation using persistent Keep-Running processes.

Reports BOTH:
  - Legality: count of games where the ML bot caused a -30 (WA/WH penalty)
  - Strength: true-win rate (score > +10, per Simulation-6 analysis criteria)
             vs +10-compensation games vs losses

Usage:
    OPENBLAS_NUM_THREADS=1 python3 eval/ml_eval.py [MODEL_NPZ] [N_GAMES]
"""

import sys, os, time
sys.path.insert(0, 'eval')
sys.path.insert(0, '.')
from run_match_kr import run_match_kr
from data.log_collector import make_wall

MODEL = sys.argv[1] if len(sys.argv) > 1 else "train/checkpoints/bc_v3_ft_weights.npz"
N     = int(sys.argv[2]) if len(sys.argv) > 2 else 40

ML  = {"cmd": f"MODEL={MODEL} python3 bot/ml_bot.py", "kr": True}
V02 = "bot/bot_submit_test"
SMP = "eval/sample_bot"

# ML always at seat 0; opponents = heuristic + 2 sample (a realistic-ish mix)
def classify(scores, seat):
    s = scores[seat]
    others = [scores[i] for i in range(4) if i != seat]
    # Illegal (WA/WH) is unambiguous: offender -30 AND all others exactly +10.
    if s == -30 and all(o == 10 for o in others):  return "ILLEGAL"
    if s == 10 and -30 in scores:                  return "COMPENSATION"
    if s > 10:                                     return "TRUE_WIN"     # real win
    if s > 0:                                      return "minor_plus"
    if s == 0:                                     return "draw"
    if s == -30:                                   return "big_dealin"   # legit heavy loss
    return "loss"

def main():
    print(f"Model: {MODEL}   Games: {N}")
    cats = {}
    net = 0
    t0 = time.time()
    for g in range(N):
        wall = make_wall(5000 + g)
        try:
            r = run_match_kr([ML, V02, SMP, SMP], wall_json=wall, quan=0, timeout=8)
        except Exception as e:
            print(f"  game {g}: error {e}"); continue
        c = classify(r["scores"], 0)
        cats[c] = cats.get(c, 0) + 1
        net += r["scores"][0]
        if c in ("ILLEGAL", "TRUE_WIN", "big_dealin"):
            print(f"  game {g}: {c}  scores={r['scores']}", flush=True)

    dt = time.time() - t0
    print(f"\n=== {N} games in {dt:.0f}s ({dt/N:.1f}s/game) ===")
    for k in ["TRUE_WIN","COMPENSATION","minor_plus","draw","loss","big_dealin","ILLEGAL"]:
        if k in cats:
            print(f"  {k:14s}: {cats[k]:3d}  ({100*cats[k]/N:.1f}%)")
    print(f"  ML net score : {net:+d}  (avg {net/N:+.1f}/game)")
    illegal = cats.get("ILLEGAL", 0)
    print(f"\nLegality: {'PASS (0 illegal)' if illegal==0 else f'FAIL ({illegal} illegal!)'}")

if __name__ == "__main__":
    main()
