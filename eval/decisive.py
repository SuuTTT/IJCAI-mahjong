"""Decisive test: does the ONE-SHOT path (Botzone mode) make the same illegal
move as the KEEP-RUNNING path? Reproduce the failing keep-running game, then for
each of the illegal seat's decision turns, replay the history through a fresh
one-shot bot and compare responses."""
import sys, os, json, subprocess
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from run_match_kr import run_match_kr
from data.log_collector import make_wall

M = "train/checkpoints/bc_v3_ft_fp16.npz"
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7019
bots = [{"cmd": f"MODEL={M} python3 bot/ml_bot.py", "kr": True} for _ in range(4)]
r = run_match_kr(bots, wall_json=make_wall(seed), quan=0, timeout=8, return_log=True)
seat = [s for s in range(4) if r['scores'][s] == -30 and sum(1 for x in r['scores'] if x == 10) == 3]
if not seat:
    print(f"no illegal this run (nondeterministic); scores {r['scores']}"); sys.exit()
seat = seat[0]
reqs = [x[0] for x in r['streams'][seat]]
resps = [x[1] for x in r['streams'][seat]]
print(f"seat {seat} illegal; {len(reqs)} turns. Comparing keep-running vs one-shot...")
mism = 0
for k in range(1, len(reqs)):
    if reqs[k].split()[0] not in ("2", "3"):
        continue
    inp = {"requests": reqs[:k+1], "responses": resps[:k]}
    p = subprocess.run(f"MODEL={M} python3 bot/ml_bot.py", shell=True,
                       input=json.dumps(inp), capture_output=True, text=True, timeout=15)
    try:
        os_resp = json.loads(p.stdout)["response"]
    except Exception:
        os_resp = "(parse-fail:" + p.stdout[:40] + ")"
    if os_resp != resps[k]:
        mism += 1
        if mism <= 6:
            print(f"  turn {k} req={reqs[k]!r}  keep-running={resps[k]!r}  ONE-SHOT={os_resp!r}")
print(f"=> {mism} mismatches over {len(reqs)} turns "
      f"({'ONE-SHOT DIFFERS (deployed bot may avoid the bug)' if mism else 'identical'})")
