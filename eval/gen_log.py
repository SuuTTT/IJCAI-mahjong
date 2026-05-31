"""
gen_log.py — run a match through the official judge and write a Botzone-format
log (JSON array alternating judge-output / bot-responses, incl. display.canHu),
so logs are directly comparable to ones downloaded from Botzone.

Usage:
    OPENBLAS_NUM_THREADS=1 python3 eval/gen_log.py OUT.log SEED BOT0 BOT1 BOT2 BOT3
where each BOTi is either a path/command, or a model alias:
    tiny | v3 | v2 | cpp | sample
"""
import sys, os, json
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from run_match_kr import call_judge, make_bot
from data.log_collector import make_wall

ALIAS = {
    "tiny":   {"cmd": "MODEL=train/checkpoints/bc_tiny_fp16.npz python3 bot/ml_bot.py", "kr": True},
    "v3":     {"cmd": "MODEL=train/checkpoints/bc_v3_ft_fp16.npz python3 bot/ml_bot.py", "kr": True},
    "v3f":    {"cmd": "MODEL=train/checkpoints/bc_v3_ft_weights.npz python3 bot/ml_bot.py", "kr": True},
    "v2":     {"cmd": "MODEL=train/checkpoints/bc_v2_weights.npz python3 bot/ml_bot.py", "kr": True},
    "cpp":    "bot/bot_submit_test",
    "sample": "eval/sample_bot",
}

def resolve(tok):
    return ALIAS.get(tok, tok)

def main():
    out, seed = sys.argv[1], int(sys.argv[2])
    specs = [resolve(t) for t in sys.argv[3:7]]
    bots = [make_bot(s, 8.0) for s in specs]
    log_entries = []
    initdata = {"walltiles": make_wall(seed), "quan": 0}

    try:
        jout = call_judge(initdata, [])
        initdata = jout.get("initdata", initdata)
        log = []
        def step(content):
            resp = {}
            for pid in range(4):
                r = bots[pid].ask_raw(content[str(pid)])
                resp[str(pid)] = {"keep_running": False, "memory": 0, "time": 0,
                                  "verdict": "OK", "response": r}
            return resp

        content = jout["content"]
        log_entries.append({"output": jout})
        resp = step(content)
        log_entries.append(resp)
        log.append(content)
        log.append({s: {"verdict": "OK", "response": resp[s]["response"]} for s in resp})

        for _ in range(400):
            jout = call_judge(initdata, log)
            log_entries.append({"output": jout})
            if jout.get("command") == "finish":
                scores = [jout["content"].get(str(i), 0) for i in range(4)]
                log_entries.append({"final_scores": scores})
                break
            content = jout["content"]
            resp = step(content)
            log_entries.append(resp)
            log.append(content)
            log.append({s: {"verdict": "OK", "response": resp[s]["response"]} for s in resp})
    finally:
        for b in bots:
            b.close()

    json.dump(log_entries, open(out, "w"), indent=1)
    # quick summary
    mx = [-99]*4
    for e in log_entries:
        if "output" in e:
            ch = e["output"].get("display", {}).get("canHu")
            if isinstance(ch, list):
                for s in range(4):
                    if isinstance(ch[s], int): mx[s] = max(mx[s], ch[s])
    fin = next((e["final_scores"] for e in log_entries if "final_scores" in e), None)
    print(f"wrote {out}  ({len(log_entries)} entries)")
    print(f"max canHu fan per seat: {mx}   final scores: {fin}")

if __name__ == "__main__":
    main()
