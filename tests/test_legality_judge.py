"""Real-judge legality regression test: ml_bot must produce 0 illegal (-30
[-30,10,10,10] pattern) over a batch of judge-run games. Run:
    OPENBLAS_NUM_THREADS=1 python3 tests/test_legality_judge.py [N]
"""
import sys, os
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from run_match_kr import run_match_kr
from data.log_collector import make_wall

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
ML = {"cmd": "MODEL=train/checkpoints/bc_v3_ft_weights.npz python3 bot/ml_bot.py", "kr": True}
SMP = "eval/sample_bot"; V02 = "bot/bot_submit_test"
cfgs = [[ML,V02,SMP,SMP],[V02,ML,SMP,SMP],[SMP,SMP,ML,V02],[ML,ML,ML,ML]]
illegal = 0
for g in range(N):
    cfg = cfgs[g % len(cfgs)]
    seats = [i for i,b in enumerate(cfg) if b is ML]
    r = run_match_kr(cfg, wall_json=make_wall(7000+g), quan=0, timeout=8)
    sc = r["scores"]
    for s in seats:
        if sc[s] == -30 and all(sc[j]==10 for j in range(4) if j!=s):
            illegal += 1
            print(f"  game {g} seat {s}: ILLEGAL {sc}")
print(f"{N} games: {illegal} illegal")
sys.exit(0 if illegal == 0 else 1)
