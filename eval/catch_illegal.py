"""
Run games until ml_bot (seat 0) hits a -30, then replay its EXACT request
stream through a lone debug ml_bot to find the offending action.
"""
import sys, subprocess
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
from run_match_kr import run_match_kr
from data.log_collector import make_wall

MODEL = "train/checkpoints/bc_v3_ft_weights.npz"
ML  = {"cmd": f"MODEL={MODEL} python3 bot/ml_bot.py", "kr": True}
SMP = "eval/sample_bot"; V02 = "bot/bot_submit_test"

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
for g in range(N):
    r = run_match_kr([ML, V02, SMP, SMP], wall_json=make_wall(9000+g),
                     quan=0, timeout=8, return_log=True)
    sc = r["scores"]
    # true illegal = offender -30 AND all others exactly +10
    if sc[0] == -30 and all(sc[i] == 10 for i in (1, 2, 3)):
        print(f"CAUGHT at game {g} (seed {9000+g}); scores={sc}")
        seat0 = r["streams"][0]
        # Replay seat-0 requests through a fresh debug ml_bot
        p = subprocess.Popen(
            f"MODEL={MODEL} ML_DEBUG=/tmp/catch.log python3 bot/ml_bot.py",
            shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)
        open('/tmp/catch.log','w').close()
        p.stdin.write("1\n"); p.stdin.flush()
        last = None
        for (req, resp) in seat0:
            p.stdin.write(req + "\n"); p.stdin.flush()
            out = []
            while True:
                line = p.stdout.readline().rstrip("\r\n")
                if line == ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<": break
                if line: out.append(line)
            my = out[0] if out else "PASS"
            last = (req, my, resp)
        p.terminate()
        print("Last request/our-response/recorded:", last)
        with open('/tmp/seat0_stream.txt', 'w') as f:
            for (req, resp) in seat0:
                f.write(f"{req}\t{resp}\n")
        print("Full seat0 stream -> /tmp/seat0_stream.txt")
        print("--- last 10 ---")
        for x in seat0[-10:]:
            print("  ", x)
        break
else:
    print(f"No illegal in {N} games.")
