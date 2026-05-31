"""
run_match_kr.py — like run_match but supports persistent Keep-Running bots.

A bot spec is either:
  - a plain string command (one-shot JSON protocol, spawned per turn — for C++ bots)
  - {"cmd": "...", "kr": True} (Keep Running: one persistent process per game)

Keep Running bots load heavy models once per game instead of per turn,
making ML-bot evaluation ~100x faster and matching real Botzone behavior.
"""

import json
import os
import subprocess
import sys

JUDGE = os.environ.get("MAHJONG_JUDGE",
                       "/workspace/Chinese-Standard-Mahjong/judge/judge")
SENTINEL = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"


def call_judge(initdata, log):
    payload = json.dumps({"initdata": initdata, "log": log})
    r = subprocess.run([JUDGE], input=payload, capture_output=True,
                       text=True, timeout=10)
    if r.returncode != 0:
        raise RuntimeError(f"Judge crashed: {r.stderr[:300]}")
    return json.loads(r.stdout)


class OneShotBot:
    """Spawns a fresh process per turn, JSON protocol."""
    def __init__(self, cmd, timeout):
        self.cmd = cmd
        self.timeout = timeout
        self.reqs = []
        self.resps = []

    def ask_raw(self, raw_request):
        self.reqs.append(raw_request)
        payload = json.dumps({"requests": self.reqs, "responses": self.resps})
        try:
            r = subprocess.run(self.cmd, shell=True, input=payload,
                               capture_output=True, text=True, timeout=self.timeout)
            resp = json.loads(r.stdout).get("response", "PASS") if r.returncode == 0 else "PASS"
        except Exception:
            resp = "PASS"
        self.resps.append(resp)
        return resp

    def close(self):
        pass


class KeepRunningBot:
    """One persistent process per game, raw-line protocol."""
    def __init__(self, cmd, timeout):
        self.timeout = timeout
        self.proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1)
        self.proc.stdin.write("1\n")  # startup handshake
        self.proc.stdin.flush()

    def ask_raw(self, raw_request):
        try:
            self.proc.stdin.write(raw_request + "\n")
            self.proc.stdin.flush()
            lines = []
            while True:
                line = self.proc.stdout.readline()
                if line == "":
                    return "PASS"
                line = line.rstrip("\r\n")
                if line == SENTINEL:
                    break
                if line:
                    lines.append(line)
            # First line that looks like an action
            for l in lines:
                w = l.split()[0] if l.split() else ""
                if w in ("PASS", "HU", "PLAY", "PENG", "CHI", "GANG", "BUGANG"):
                    return l
            return lines[0] if lines else "PASS"
        except Exception:
            return "PASS"

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            try: self.proc.kill()
            except Exception: pass


def make_bot(spec, timeout):
    if isinstance(spec, dict) and spec.get("kr"):
        return KeepRunningBot(spec["cmd"], timeout)
    cmd = spec["cmd"] if isinstance(spec, dict) else spec
    return OneShotBot(cmd, timeout)


def run_match_kr(bot_specs, wall_json="", quan=0, timeout=8.0, return_log=False):
    assert len(bot_specs) == 4
    bots = [make_bot(s, timeout) for s in bot_specs]
    log = []
    # record per-seat (request, response) streams for deterministic replay
    streams = [[] for _ in range(4)]
    initdata = {}
    if wall_json: initdata["walltiles"] = wall_json
    if 0 <= quan <= 3: initdata["quan"] = quan

    try:
        jout = call_judge(initdata, [])
        initdata = jout.get("initdata", {})
        content = jout["content"]

        _final = {"display": None}
        def out(scores, winner):
            res = {"scores": scores, "winner": winner, "quan": initdata.get("quan", 0)}
            res["display"] = _final["display"]
            if return_log:
                res["streams"] = streams
            return res

        # init broadcast
        resp = {}
        for pid in range(4):
            req = content[str(pid)]
            r = bots[pid].ask_raw(req)
            streams[pid].append((req, r))
            resp[str(pid)] = {"verdict": "OK", "response": r}
        log.append(content); log.append(resp)

        for _ in range(400):
            jout = call_judge(initdata, log)
            if jout.get("command") == "finish":
                scores = [jout["content"].get(str(i), 0) for i in range(4)]
                winner = next((i for i, s in enumerate(scores) if s > 0), -1)
                _final["display"] = jout.get("display")
                return out(scores, winner)
            content = jout["content"]
            resp = {}
            for pid in range(4):
                req = content[str(pid)]
                r = bots[pid].ask_raw(req)
                streams[pid].append((req, r))
                resp[str(pid)] = {"verdict": "OK", "response": r}
            log.append(content); log.append(resp)

        return out([0, 0, 0, 0], -1)
    finally:
        for b in bots:
            b.close()


if __name__ == "__main__":
    # Smoke test: ml_bot (KR) vs heuristic vs 2x sample
    import time
    ML = {"cmd": "MODEL=train/checkpoints/bc_v3_ft_weights.npz python3 bot/ml_bot.py", "kr": True}
    V02 = "bot/bot_submit_test"
    SMP = "eval/sample_bot"
    t0 = time.time()
    r = run_match_kr([ML, V02, SMP, SMP], timeout=8)
    print(f"scores={r['scores']} winner={r['winner']}  ({time.time()-t0:.1f}s)")
