"""Quick head-to-head eval: ML-bc_v2 vs v0.2-heuristic vs 2x sample."""
import sys, time
sys.path.insert(0, 'eval')
from run_match import run_match

ML  = 'MODEL=train/checkpoints/bc_v2_weights.npz python3 bot/ml_bot.py'
V02 = 'bot/bot_submit_test'
SMP = 'eval/sample_bot'

wins = {'ML': 0, 'V02': 0, 'SMP': 0, 'h': 0}
scores = [0, 0, 0, 0]
t0 = time.time()

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
for seed in range(N):
    try:
        r = run_match([ML, V02, SMP, SMP], timeout=8)
        for i, s in enumerate(r['scores']):
            scores[i] += s
        w = r['winner']
        label = {0: 'ML', 1: 'V02'}.get(w, 'SMP' if w in (2, 3) else 'h')
        wins[label] += 1
        print(f'seed {seed}: {r["scores"]}  winner={label}', flush=True)
    except Exception as e:
        print(f'seed {seed}: error {e}')

elapsed = time.time() - t0
print(f'\n--- {elapsed:.0f}s for {N} games ({elapsed/N:.1f}s/game) ---')
print(f'Wins: {wins}')
print(f'AvgScore: ML={scores[0]/N:+.1f}  V02={scores[1]/N:+.1f}  SMP={scores[2]/N:+.1f}')
