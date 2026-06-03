"""
gate_candidate.py — decide whether a candidate model should replace the reference (r18).
Runs the contest-relevant battery established this session:
  1) self-play DRAW RATE of the candidate vs itself  (conversion proxy: lower = builds more
     8-fan hands; r18 baseline ~89%). The cleanest non-opponent-specific signal.
  2) h2h_fast candidate vs reference (must NOT regress beyond the ~±400/1200g noise floor).
  3) farming net vs a panel of weak opponents (bc_v3_ft, ppo_vb, bc_tiny) — must improve on
     BALANCE (not just one opponent — that was the gen2/bc_v3_ft overfit trap).

Usage: OPENBLAS_NUM_THREADS=1 python3 eval/gate_candidate.py CAND.npz REF.npz [N]
"""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def run(args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, "OPENBLAS_NUM_THREADS": "1"}).stdout

if __name__ == "__main__":
    cand, ref = sys.argv[1], sys.argv[2]
    N = sys.argv[3] if len(sys.argv) > 3 else "2500"
    py = sys.executable
    print(f"=== GATE: cand={os.path.basename(cand)}  ref={os.path.basename(ref)} ===\n")
    print("[1] candidate self-play draw rate (conversion proxy; r18~89%):")
    print(run([py, "eval/outcome_stats.py", cand, cand, "2500", "12"]))
    print("[1b] reference self-play draw rate (baseline):")
    print(run([py, "eval/outcome_stats.py", ref, ref, "2500", "12"]))
    print("[2] h2h candidate vs reference (regression check, noise floor ~±400/1200g):")
    print(run([py, "eval/h2h_fast.py", cand, ref, "1500", "12"]))
    print("[3] farming panel (must improve on balance, not one opponent):")
    for opp in ["bc_v3_ft_fp16", "ppo_vb_fp16", "bc_tiny_fp16"]:
        op = f"train/checkpoints/{opp}.npz"
        if os.path.exists(op):
            print(f"  -- cand vs {opp} --")
            print(run([py, "eval/outcome_stats.py", cand, op, N, "12"]))
            print(f"  -- ref  vs {opp} --")
            print(run([py, "eval/outcome_stats.py", ref, op, N, "12"]))
