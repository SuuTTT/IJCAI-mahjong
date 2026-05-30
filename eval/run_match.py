"""
run_match.py — drive a single 4-player game through the official judge.

Usage:
    python3 run_match.py bot0 bot1 bot2 bot3 [--wall WALL_JSON] [--quan N] [--timeout 5]

Each botN is a shell command that accepts Botzone JSON on stdin and
prints {"response": "..."} to stdout (the one-shot JSON protocol).

Returns a dict: {"scores": [s0,s1,s2,s3], "turns": N, "winner": pid|-1}
"""

import argparse
import json
import subprocess
import sys
import os
import time

JUDGE = os.environ.get(
    "MAHJONG_JUDGE",
    "/workspace/Chinese-Standard-Mahjong/judge/judge",
)

# ── helpers ────────────────────────────────────────────────────────────────────

def call_judge(initdata: dict, log: list) -> dict:
    payload = json.dumps({"initdata": initdata, "log": log})
    r = subprocess.run(
        [JUDGE],
        input=payload,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Judge crashed:\n{r.stderr[:400]}")
    return json.loads(r.stdout)


def call_bot(cmd: str, requests: list, responses: list, timeout: float) -> str:
    payload = json.dumps({"requests": requests, "responses": responses})
    try:
        r = subprocess.run(
            cmd,
            shell=True,
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            return "PASS"
        data = json.loads(r.stdout)
        return data.get("response", "PASS")
    except subprocess.TimeoutExpired:
        return "PASS"
    except Exception:
        return "PASS"


# ── main game loop ─────────────────────────────────────────────────────────────

def run_match(
    bot_cmds: list,          # 4 shell commands
    wall_json: str = "",     # pre-fixed walltiles string, or "" for random
    quan: int = -1,          # -1 = random
    timeout: float = 5.0,
    verbose: bool = False,
) -> dict:

    assert len(bot_cmds) == 4

    # Per-player conversation histories
    histories = [{"requests": [], "responses": []} for _ in range(4)]
    log = []

    # ── Round 0: get initdata + first broadcast ────────────────────────────────
    initdata = {}
    if wall_json:
        initdata["walltiles"] = wall_json
    if 0 <= quan <= 3:
        initdata["quan"] = quan

    jout = call_judge(initdata, log=[])
    # jout["command"] == "request", content[i] = "0 i quan"
    if jout.get("command") != "request":
        raise RuntimeError(f"Unexpected judge output on init: {jout}")

    initdata = jout.get("initdata", {})
    content = jout["content"]

    # All 4 players receive init notification; all must reply PASS
    for pid in range(4):
        histories[pid]["requests"].append(content[str(pid)])

    bot_responses = {}
    for pid in range(4):
        resp = call_bot(
            bot_cmds[pid],
            histories[pid]["requests"],
            histories[pid]["responses"],
            timeout,
        )
        histories[pid]["responses"].append(resp)
        bot_responses[str(pid)] = {"verdict": "OK", "response": resp}

    log.append(content)           # judge output (log[0])
    log.append(bot_responses)     # player responses (log[1])

    # ── Game loop ──────────────────────────────────────────────────────────────
    turn = 0
    MAX_TURNS = 400

    while turn < MAX_TURNS:
        jout = call_judge(initdata, log)

        if verbose:
            action = jout.get("display", {}).get("action", "?")
            print(f"  turn={turn} action={action}", file=sys.stderr)

        cmd = jout.get("command")
        if cmd == "finish":
            scores = [jout["content"].get(str(i), 0) for i in range(4)]
            winner = -1
            for i, s in enumerate(scores):
                if s > 0:
                    winner = i
            return {
                "scores": scores,
                "turns": turn,
                "winner": winner,
                "quan": initdata.get("quan", 0),
            }

        if cmd != "request":
            raise RuntimeError(f"Unknown judge command: {cmd}")

        content = jout["content"]
        bot_responses = {}

        for pid in range(4):
            req = content[str(pid)]
            histories[pid]["requests"].append(req)
            resp = call_bot(
                bot_cmds[pid],
                histories[pid]["requests"],
                histories[pid]["responses"],
                timeout,
            )
            histories[pid]["responses"].append(resp)
            bot_responses[str(pid)] = {"verdict": "OK", "response": resp}

        log.append(content)
        log.append(bot_responses)
        turn += 1

    return {"scores": [0, 0, 0, 0], "turns": turn, "winner": -1, "quan": initdata.get("quan", 0)}


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("bots", nargs=4, metavar="BOT")
    p.add_argument("--quan", type=int, default=-1)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    result = run_match(args.bots, quan=args.quan, timeout=args.timeout, verbose=args.verbose)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
