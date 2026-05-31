"""Reproduce a seat-0 WA and dump the EXACT rejected action + judge display."""
import sys, json, subprocess
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
from run_match_kr import run_match_kr
from data.log_collector import make_wall

seed = int(sys.argv[1])
MODEL = "train/checkpoints/bc_v3_ft_weights.npz"
for attempt in range(40):
    open('/tmp/diag2.log', 'w').close()
    ML = {"cmd": f"MODEL={MODEL} ML_DEBUG=/tmp/diag2.log python3 bot/ml_bot.py", "kr": True}
    r = run_match_kr([ML, "bot/bot_submit_test", "eval/sample_bot", "eval/sample_bot"],
                     wall_json=make_wall(seed), quan=0, timeout=8, return_log=True)
    sc = r["scores"]
    if sc[0] == -30 and all(sc[i] == 10 for i in (1, 2, 3)):
        disp = r["display"]
        print(f"WA reproduced (attempt {attempt}). display={json.dumps(disp)}")
        # dump my last decision and surrounding context
        lines = [l for l in open('/tmp/diag2.log')]
        print("--- my last 3 decisions ---")
        for l in [x for x in lines if x.startswith("req=")][-3:]:
            print("  ", l.strip())
        # dump seat-0 raw stream tail
        print("--- seat0 raw stream (last 6) ---")
        for x in r["streams"][0][-6:]:
            print("  ", x)
        break
else:
    print("no WA in 40 attempts")
