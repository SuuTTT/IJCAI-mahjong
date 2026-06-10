"""
run_gauntlet.py — clean gauntlet driver (replaces buggy gredux2.sh bash arithmetic).
Runs a candidate vs each of 6 strong top-30 imitations through the fixed bench, parses the
summary 'net=' line in Python, sums. Usage: python3 run_gauntlet.py <CAND> <CPKL>
"""
import subprocess, sys, os, re

CAND, CPKL = sys.argv[1], sys.argv[2]
OPPS = ["qwqwqawawa", "dimaria", "渡鸦", "knight", "ChloePrice", "QiuQiuR"]
N = int(os.environ.get("GN", "12"))
base = "/root/mahjong"
CD, OD = f"rc3_{CAND}", f"ro3_{CAND}"
os.system(f"cd {base} && rm -rf {CD} {OD} && cp -r botA {CD} && cp -r botA {OD} && mkdir -p {CD}/data {OD}/data")
os.system(f"ln -sf {CPKL} {base}/{CD}/data/cnn.pkl")
BC = "CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 BOTZONE_JSON=0 python3 __main__.py"
CBC = (os.environ.get("CAND_ENV", "") + " " + BC).strip()   # candidate-only extra env (e.g. CAIEST_QNET=...)
env = dict(os.environ, MAHJONG_JUDGE=f"{base}/Chinese-Standard-Mahjong/judge/judge",
           BENCH_TIMEOUT=os.environ.get("BENCH_TIMEOUT", "40"), WALL_SEED_BASE="600000")
total, played_all, stuck_all = 0, 0, 0
print(f"=== gauntlet {CAND} START ===", flush=True)
for opp in OPPS:
    os.system(f"ln -sf {base}/ckpt/g30_{opp}.pkl {base}/{OD}/data/cnn.pkl")
    cmd = ["python3", "eval/bench_vs_bot.py", f"cd {base}/{CD} && {CBC}",
           f"cd {base}/{OD} && {BC}", str(N), CAND, opp]
    r = subprocess.run(cmd, cwd=base, env=env, capture_output=True, text=True)
    out = r.stdout
    m = re.search(re.escape(CAND) + r": net=([-+]?\d+)", out)
    mp = re.search(r"(\d+) played", out)
    ms = re.search(r"(\d+) stuck", out)
    net = int(m.group(1)) if m else None
    played = int(mp.group(1)) if mp else 0
    st = int(ms.group(1)) if ms else 0
    if net is not None:
        total += net; played_all += played; stuck_all += st
    print(f"  vs {opp}: net={net} played={played} stuck={st}", flush=True)
print(f"=== gauntlet {CAND} TOTAL net={total} (played={played_all} stuck={stuck_all}) ===", flush=True)
