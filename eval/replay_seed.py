"""Replay one wall seed with ml_bot debug logging to find illegal moves."""
import sys, os
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
from run_match_kr import run_match_kr
from data.log_collector import make_wall

seed  = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
model = sys.argv[2] if len(sys.argv) > 2 else "train/checkpoints/bc_v3_ft_weights.npz"

ML = {"cmd": f"MODEL={model} ML_DEBUG=1 python3 bot/ml_bot.py", "kr": True}
r = run_match_kr([ML, "bot/bot_submit_test", "eval/sample_bot", "eval/sample_bot"],
                 wall_json=make_wall(seed), quan=0, timeout=8)
print("RESULT scores=", r["scores"], "winner=", r["winner"])
