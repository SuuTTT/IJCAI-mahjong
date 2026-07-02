"""preflight.py — MANDATORY invariant checks before trusting any gate campaign.
Run: python3 preflight.py [--full].  Exits non-zero on any failure. (best-practice fix 2026-07-02)
Invariants:
  I1 self-Play request2obs returns None (documented trap that silently killed E8 discard lookahead)
  I2 _obs() renders a valid post-discard observation (the E14 fix path works)
  I3 calibration: any gate cand==ref lam=0/knob-off must equal 2.500 exactly (spot 40 seeds)
  I4 mechanism engagement: e14 value overlay hooked_rate must be > 0.9 (spot 8 seeds)
"""
import sys, json, subprocess, numpy as np
sys.path.insert(0, ".")
from feature import FeatureAgent
fails = []
# --- I1 + I2: replay a short match prefix ---
ag = [FeatureAgent(i) for i in range(4)]
lines = []
for ln in open("data/data.txt", encoding="UTF-8"):
    lines.append(ln.strip())
    if len(lines) > 400: break
start = lines.index([l for l in lines if l.startswith("Match")][0])
i1 = i2 = None
for ln in lines[start + 1:]:
    t = ln.split()
    if not t or t[0] == "Match": break
    if t[0] == "Wind":
        for a in ag: a.request2obs(ln)
    elif t[0] == "Player":
        p = int(t[1])
        if t[2] == "Deal": ag[p].request2obs(" ".join(t[2:]))
        elif t[2] == "Draw":
            for i in range(4):
                ag[i].request2obs(" ".join(t[2:]) if i == p else " ".join(t[:3]))
        elif t[2] == "Play":
            r = ag[p].request2obs(ln)
            i1 = (r is None)
            o = ag[p]._obs()["observation"]
            i2 = bool(o.shape == (38, 4, 9) and o.sum() > 0)
            break
if i1 is not True: fails.append("I1 self-Play None contract changed — audit every request2obs caller")
if i2 is not True: fails.append("I2 _obs() post-discard render broken")
print(f"I1 self-Play returns None: {i1}\nI2 _obs() renders post state: {i2}")
if "--full" in sys.argv:
    A = "ckpt/aug/aug_128x40_s0.pkl"
    r = subprocess.run(["python3", "e14_gate.py", "--cand", A, "--ref", A, "--lam", "0",
                        "--seeds", "40", "--workers", "40", "--seed0", "990000", "--out", "/tmp/pf_calib.json"],
                       capture_output=True, text=True, timeout=900)
    c = json.load(open("/tmp/pf_calib.json"))
    pts = c["placement_pts"]
    ok3 = abs(pts - 2.5) < 1e-9
    print(f"I3 calibration 2.500 exact: {ok3} ({pts})")
    if not ok3: fails.append("I3 calibration broken")
    r = subprocess.run(["python3", "e14_gate.py", "--cand", A, "--ref", A, "--lam", "0.5",
                        "--value", "ckpt/value_256x40.pkl",
                        "--seeds", "8", "--workers", "8", "--seed0", "991000", "--out", "/tmp/pf_hook.json"],
                       capture_output=True, text=True, timeout=900)
    h = json.load(open("/tmp/pf_hook.json"))
    hr = h.get("hooked_rate", 0)
    ok4 = hr > 0.9
    print(f"I4 mechanism hooked_rate>0.9: {ok4} ({hr})")
    if not ok4: fails.append("I4 overlay mechanism not engaging")
if fails:
    print("PREFLIGHT FAIL:\n  " + "\n  ".join(fails)); sys.exit(1)
print("PREFLIGHT PASS")
