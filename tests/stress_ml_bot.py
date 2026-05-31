"""
stress_ml_bot.py — run ml_bot through MANY real judge games and assert
it NEVER triggers a -30 penalty (WA illegal action / WH wrong-hu).

Per the Simulation-6 analysis: a single -30 is catastrophic (failure
compensation dominates the weak field). The bar is ZERO illegal moves.

Usage:
    MODEL=train/checkpoints/bc_v3_ft_weights.npz python3 tests/stress_ml_bot.py [N_GAMES]
"""

import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))

from data.log_collector import run_match_with_log, make_wall

MODEL = os.environ.get("MODEL", "train/checkpoints/bc_v3_ft_weights.npz")
ML  = f'MODEL={MODEL} python3 bot/ml_bot.py'
SMP = 'eval/sample_bot'
V02 = 'bot/bot_submit_test'


def detect_minus30(scores):
    """A -30 in scores means someone got a WA/WH penalty."""
    return [i for i, s in enumerate(scores) if s == -30]


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    # Put ml_bot in all 4 seats across games so every code path is exercised.
    configs = [
        [ML, ML, ML, ML],     # ml self-play — exercises all roles
        [ML, V02, SMP, SMP],
        [V02, ML, SMP, SMP],
        [SMP, SMP, ML, V02],
    ]

    ml_failures = 0
    total_minus30 = 0
    games_run = 0

    print(f"Stress-testing ml_bot ({MODEL}) over {n_games} games...")
    for g in range(n_games):
        cfg = configs[g % len(configs)]
        ml_seats = [i for i, b in enumerate(cfg) if b == ML]
        wall = make_wall(1000 + g)
        try:
            r = run_match_with_log(cfg, wall_json=wall, quan=0, timeout=10)
        except Exception as e:
            print(f"  game {g}: harness error {e}")
            continue
        games_run += 1
        scores = r["scores"]
        bad = detect_minus30(scores)
        if bad:
            total_minus30 += 1
            ml_at_fault = [i for i in bad if i in ml_seats]
            if ml_at_fault:
                ml_failures += 1
                print(f"  game {g}: ML at seat {ml_at_fault} got -30!  scores={scores}")
            # else: another bot failed, ml just got +10 compensation
        if (g + 1) % 10 == 0:
            print(f"  {g+1}/{n_games}  ml_failures={ml_failures}  total_-30_games={total_minus30}",
                  flush=True)

    print(f"\n=== Results over {games_run} games ===")
    print(f"Games with any -30:      {total_minus30}")
    print(f"Games where ML caused -30: {ml_failures}")
    if ml_failures == 0:
        print("PASS — ml_bot never triggered an illegal-move penalty.")
        return 0
    else:
        print(f"FAIL — ml_bot caused {ml_failures} illegal-move penalties.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
