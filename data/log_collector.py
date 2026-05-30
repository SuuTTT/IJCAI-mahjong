"""
log_collector.py — run many games via run_match.py, save logs as JSONL.

Each output line is a JSON object containing the full game log +
per-player action sequences, usable for SL data extraction.

Usage:
    python3 data/log_collector.py \
        --bot0 "bot/bot_submit_test" \
        --bot1 "eval/sample_bot" \
        --games 500 --workers 4 \
        --out data/raw/selfplay.jsonl
"""

import argparse
import json
import os
import sys
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval.run_match import run_match, call_judge

SUITS  = list("WBT")
HONORS = [f"F{i}" for i in range(1,5)] + [f"J{i}" for i in range(1,4)]


def make_wall(seed: int) -> str:
    tiles = [f"{s}{n}" for s in SUITS for n in range(1,10)] * 4
    tiles += [f"F{n}" for n in range(1,5)] * 4
    tiles += [f"J{n}" for n in range(1,4)] * 4
    rng = random.Random(seed)
    rng.shuffle(tiles)
    return " ".join(tiles)


def run_one(args):
    bot_cmds, seed, quan, timeout = args
    wall = make_wall(seed)
    try:
        # We need the full log, not just the scores.
        # Monkey-patch run_match to also return log.
        result = run_match_with_log(bot_cmds, wall_json=wall, quan=quan, timeout=timeout)
        result["seed"] = seed
        return result
    except Exception as e:
        return {"error": str(e), "seed": seed}


def run_match_with_log(bot_cmds, wall_json="", quan=0, timeout=5.0):
    """Like run_match but also returns the full game log."""
    import subprocess

    JUDGE = os.environ.get(
        "MAHJONG_JUDGE",
        "/workspace/Chinese-Standard-Mahjong/judge/judge",
    )

    def call_judge_local(initdata, log):
        payload = json.dumps({"initdata": initdata, "log": log})
        r = subprocess.run([JUDGE], input=payload, capture_output=True,
                           text=True, timeout=10)
        if r.returncode != 0:
            raise RuntimeError(f"Judge: {r.stderr[:200]}")
        return json.loads(r.stdout)

    def call_bot_local(cmd, requests, responses):
        payload = json.dumps({"requests": requests, "responses": responses})
        try:
            r = subprocess.run(cmd, shell=True, input=payload,
                               capture_output=True, text=True, timeout=timeout)
            if r.returncode != 0: return "PASS"
            return json.loads(r.stdout).get("response","PASS")
        except Exception:
            return "PASS"

    histories = [{"requests":[], "responses":[]} for _ in range(4)]
    log = []

    initdata = {}
    if wall_json: initdata["walltiles"] = wall_json
    if 0 <= quan <= 3: initdata["quan"] = quan

    jout = call_judge_local(initdata, [])
    initdata = jout.get("initdata", {})
    content  = jout["content"]

    for pid in range(4):
        histories[pid]["requests"].append(content[str(pid)])

    bot_responses = {}
    for pid in range(4):
        resp = call_bot_local(bot_cmds[pid],
                              histories[pid]["requests"],
                              histories[pid]["responses"])
        histories[pid]["responses"].append(resp)
        bot_responses[str(pid)] = {"verdict":"OK","response":resp}

    log.append(content)
    log.append(bot_responses)

    for _ in range(400):
        jout = call_judge_local(initdata, log)
        cmd  = jout.get("command")
        if cmd == "finish":
            scores = [jout["content"].get(str(i), 0) for i in range(4)]
            return {
                "scores": scores,
                "log": log,
                "quan": initdata.get("quan", 0),
                "walltiles": initdata.get("walltiles",""),
            }
        content = jout["content"]
        bot_responses = {}
        for pid in range(4):
            histories[pid]["requests"].append(content[str(pid)])
            resp = call_bot_local(bot_cmds[pid],
                                  histories[pid]["requests"],
                                  histories[pid]["responses"])
            histories[pid]["responses"].append(resp)
            bot_responses[str(pid)] = {"verdict":"OK","response":resp}
        log.append(content)
        log.append(bot_responses)

    return {"scores":[0]*4, "log":log, "quan":initdata.get("quan",0)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bot0", default="bot/bot_submit_test")
    p.add_argument("--bot1", default="eval/sample_bot")
    p.add_argument("--bot2", default="eval/sample_bot")
    p.add_argument("--bot3", default="eval/sample_bot")
    p.add_argument("--games",   type=int,   default=200)
    p.add_argument("--workers", type=int,   default=1)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--seed",    type=int,   default=0)
    p.add_argument("--quan",    type=int,   default=0)
    p.add_argument("--out",     default="data/raw/selfplay.jsonl")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    bot_cmds = [args.bot0, args.bot1, args.bot2, args.bot3]

    work = [(bot_cmds, args.seed + i, args.quan, args.timeout)
            for i in range(args.games)]

    written = 0
    errors  = 0
    with open(args.out, "w") as f:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for fut in as_completed(ex.submit(run_one, w) for w in work):
                    r = fut.result()
                    if "error" in r:
                        errors += 1
                    else:
                        f.write(json.dumps(r) + "\n")
                        written += 1
                    if (written + errors) % 10 == 0:
                        print(f"  {written} ok / {errors} errors", flush=True)
        else:
            for w in work:
                r = run_one(w)
                if "error" in r:
                    errors += 1
                    print(f"  error seed={r['seed']}: {r['error']}", flush=True)
                else:
                    f.write(json.dumps(r) + "\n")
                    written += 1
                if (written + errors) % 10 == 0:
                    print(f"  {written}/{args.games} done", flush=True)

    print(f"Done: {written} games saved to {args.out}, {errors} errors")


if __name__ == "__main__":
    main()
