"""
run_match_kr.py — like run_match but supports persistent Keep-Running bots.

A bot spec is either:
  - a plain string command (one-shot JSON protocol, spawned per turn — for C++ bots)
  - {"cmd": "...", "kr": True} (Keep Running: one persistent process per game)

Keep Running bots load heavy models once per game instead of per turn,
making ML-bot evaluation ~100x faster and matching real Botzone behavior.

ROBUSTNESS (2026-06-09): the KR read path now enforces self.timeout via select(), so a bot
that fails to emit the SENTINEL (protocol desync / wedged inference) no longer hangs the whole
bench forever (the prior bug: bare readline() with no timeout -> load~0 deadlock on slow boxes).
On read-timeout the bot is marked dead and the GAME is aborted (returned with stuck=True) so the
driver can skip it instead of blocking. Set BENCH_DEBUG=1 to capture bot stderr to /tmp for diagnosis.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

JUDGE = os.environ.get("MAHJONG_JUDGE",
                       "/workspace/Chinese-Standard-Mahjong/judge/judge")
SENTINEL = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"
_DEBUG = os.environ.get("BENCH_DEBUG", "") not in ("", "0")


def call_judge(initdata, log):
    payload = json.dumps({"initdata": initdata, "log": log})
    r = subprocess.run([JUDGE], input=payload, capture_output=True,
                       text=True, timeout=10)
    if r.returncode != 0:
        raise RuntimeError(f"Judge crashed: {r.stderr[:300]}")
    return json.loads(r.stdout)


class BotStuck(Exception):
    """Raised when a KR bot fails to respond within its timeout (wedged / protocol desync)."""


class OneShotBot:
    """Spawns a fresh process per turn, JSON protocol."""
    def __init__(self, cmd, timeout, label=""):
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
    def __init__(self, cmd, timeout, label=""):
        self.timeout = timeout
        self.dead = False
        self.label = label
        errdst = subprocess.DEVNULL
        if _DEBUG:
            safe = "".join(c if c.isalnum() else "_" for c in (label or "bot"))[:40]
            self._errf = open(f"/tmp/bench_err_{safe}_{os.getpid()}.log", "w")
            errdst = self._errf
        else:
            self._errf = None
        self.proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=errdst,
                                     text=True, bufsize=1)
        # Background reader thread -> queue. A blocking readline in a thread correctly drains
        # Python's userspace buffer (a select() on the raw fd does NOT — it misses readahead,
        # the bug that made every turn spuriously time out). Main thread polls the queue with
        # self.timeout, so a wedged bot can't hang the bench forever.
        self.q = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self.proc.stdin.write("1\n")  # startup handshake
        self.proc.stdin.flush()

    def _read_loop(self):
        try:
            for line in self.proc.stdout:
                self.q.put(line)
        except Exception:
            pass
        self.q.put(None)  # EOF sentinel

    def ask_raw(self, raw_request):
        if self.dead:
            return "PASS"
        try:
            self.proc.stdin.write(raw_request + "\n")
            self.proc.stdin.flush()
        except Exception:
            self.dead = True
            raise BotStuck(f"{self.label}: stdin write failed")
        lines = []
        deadline = time.time() + self.timeout
        while True:
            try:
                line = self.q.get(timeout=max(0.01, deadline - time.time()))
            except queue.Empty:
                self.dead = True
                raise BotStuck(f"{self.label}: no SENTINEL within {self.timeout}s")
            if line is None:      # EOF (bot exited)
                return "PASS"
            line = line.rstrip("\r\n")
            if line == SENTINEL:
                break
            if line:
                lines.append(line)
        for l in lines:
            w = l.split()[0] if l.split() else ""
            if w in ("PASS", "HU", "PLAY", "PENG", "CHI", "GANG", "BUGANG"):
                return l
        return lines[0] if lines else "PASS"

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            try: self.proc.kill()
            except Exception: pass
        if self._errf:
            try: self._errf.close()
            except Exception: pass


def make_bot(spec, timeout, label=""):
    if isinstance(spec, dict) and spec.get("kr"):
        return KeepRunningBot(spec["cmd"], timeout, label)
    cmd = spec["cmd"] if isinstance(spec, dict) else spec
    return OneShotBot(cmd, timeout, label)


def run_match_kr(bot_specs, wall_json="", quan=0, timeout=8.0, return_log=False, labels=None):
    assert len(bot_specs) == 4
    labels = labels or [f"seat{i}" for i in range(4)]
    bots = [make_bot(s, timeout, labels[i]) for i, s in enumerate(bot_specs)]
    log = []
    streams = [[] for _ in range(4)]
    initdata = {}
    if wall_json: initdata["walltiles"] = wall_json
    if 0 <= quan <= 3: initdata["quan"] = quan

    stuck = False
    try:
        jout = call_judge(initdata, [])
        initdata = jout.get("initdata", {})
        content = jout["content"]

        _final = {"display": None}
        def out(scores, winner):
            res = {"scores": scores, "winner": winner, "quan": initdata.get("quan", 0), "stuck": stuck}
            res["display"] = _final["display"]
            if return_log:
                res["streams"] = streams
            return res

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
    except BotStuck as e:
        stuck = True
        if _DEBUG:
            print(f"  [STUCK] {e}", file=sys.stderr, flush=True)
        return {"scores": [0, 0, 0, 0], "winner": -1, "quan": initdata.get("quan", 0),
                "stuck": True, "display": None}
    finally:
        for b in bots:
            b.close()


if __name__ == "__main__":
    import time
    ML = {"cmd": "MODEL=train/checkpoints/bc_v3_ft_weights.npz python3 bot/ml_bot.py", "kr": True}
    V02 = "bot/bot_submit_test"
    SMP = "eval/sample_bot"
    t0 = time.time()
    r = run_match_kr([ML, V02, SMP, SMP], timeout=8)
    print(f"scores={r['scores']} winner={r['winner']}  ({time.time()-t0:.1f}s)")
