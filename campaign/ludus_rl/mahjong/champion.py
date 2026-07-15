"""kdens3 champion (IJCAI-2026 Chinese Standard Mahjong runner-up) as a
Botzone-protocol agent for the Ludus engine.

Wraps the published NumPy ensemble bot (mcr_champion/__main__.py) as a persistent
"keep-running" subprocess: the 3-model fp16 ensemble loads once, then each turn
is one JSON line in -> one {"response": ...} line out. Our engine already emits
judge-exact Botzone requests, so the champion sees exactly what it saw on
Botzone. Falls back to a legal discard / PASS if the subprocess errors or times
out, so a game can never hang.
"""

import json
import os
import select
import subprocess

CHAMP_DIR = os.environ.get("MCR_CHAMPION_DIR", "/root/mcr_champion")
NPZS = os.environ.get("MCR_CHAMPION_NPZS",
                      "kdens_s0_fp16.npz,kdens_s1_fp16.npz,kdens_s2_fp16.npz")


class ChampionBot:
    def __init__(self, champ_dir=CHAMP_DIR, timeout=20.0, npzs=None):
        self.dir = champ_dir
        self.timeout = timeout
        # npzs: optional comma-separated npz list overriding the env default.
        # A single file (e.g. "kdens_s0_fp16.npz") runs ONE net -> the weaker
        # "Hard" tier; the default (None) keeps the 3-model kdens3 ensemble.
        self.npzs = npzs or NPZS
        self.requests = []
        self.responses = []
        self.proc = None
        self.hand = []        # tracked only for the safety fallback

    def _spawn(self):
        env = dict(os.environ, ENSEMBLE_NPZS=self.npzs)
        self.proc = subprocess.Popen(
            ["python3", "__main__.py"], cwd=self.dir, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def _track(self, req):
        p = req.split()
        if p and p[0] == "1":
            self.hand = list(p[5:5 + 13])
        elif p and p[0] == "2":
            self.hand.append(p[1])

    def _fallback(self, req):
        p = req.split()
        if p and p[0] == "2" and self.hand:      # our draw -> discard a held tile
            return f"PLAY {self.hand[-1]}"
        return "PASS"

    def respond(self, request):
        self._track(request)
        self.requests.append(request)
        resp = None
        try:
            if self.proc is None or self.proc.poll() is not None:
                self._spawn()
            self.proc.stdin.write(json.dumps(
                {"requests": self.requests, "responses": self.responses}) + "\n")
            self.proc.stdin.flush()
            r, _, _ = select.select([self.proc.stdout], [], [], self.timeout)
            if r:
                line = self.proc.stdout.readline()
                if line:
                    resp = (json.loads(line).get("response") or "").strip()
        except Exception:
            resp = None
        if not resp:
            resp = self._fallback(request)
        # keep the fallback hand loosely in sync on our own discards
        pp = resp.split()
        if pp and pp[0] == "PLAY" and pp[1] in self.hand:
            self.hand.remove(pp[1])
        self.responses.append(resp)
        return resp

    def close(self):
        if self.proc:
            try:
                self.proc.stdin.close()
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None

    def __del__(self):
        self.close()
