import sys, json
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
from run_match_kr import run_match_kr
from data.log_collector import make_wall
seed = int(sys.argv[1])
ML = {"cmd": "MODEL=train/checkpoints/bc_v3_ft_weights.npz python3 bot/ml_bot.py", "kr": True}
for _ in range(20):
    r = run_match_kr([ML, "bot/bot_submit_test", "eval/sample_bot", "eval/sample_bot"],
                     wall_json=make_wall(seed), quan=0, timeout=8)
    sc = r["scores"]
    if sc[0] == -30 and all(sc[i] == 10 for i in (1, 2, 3)):
        print("ILLEGAL reproduced. display=", json.dumps(r["display"]))
        break
else:
    print("did not reproduce in 20 tries (stochastic opponents)")
