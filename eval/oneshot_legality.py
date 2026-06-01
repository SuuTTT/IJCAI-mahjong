"""One-shot (Botzone-mode) legality test. Legality is model-independent, so we
use the TINY model for fast per-turn loads. 4x one-shot self-play through the
official judge; count true illegal ([-30,10,10,10]) games."""
import sys, os
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from run_match import run_match
from data.log_collector import make_wall

M = sys.argv[1] if len(sys.argv) > 1 else "train/checkpoints/bc_tiny_fp16.npz"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
cmd = f"MODEL={M} python3 bot/ml_bot.py"
ill = 0
for g in range(N):
    r = run_match([cmd, cmd, cmd, cmd], wall_json=make_wall(7000 + g), quan=0, timeout=15)
    sc = r['scores']
    if any(sc[i] == -30 and sum(1 for x in sc if x == 10) == 3 for i in range(4)):
        ill += 1
        print(f"  seed {7000+g}: ILLEGAL {sc}", flush=True)
    if (g + 1) % 5 == 0:
        print(f"  {g+1}/{N} done, {ill} illegal", flush=True)
print(f"ONE-SHOT self-play (Botzone mode): {N} games, {ill} illegal")
