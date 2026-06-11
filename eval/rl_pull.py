"""
rl_pull.py — poll ssh8's rl_league.log, parse per-iter metrics + snapshot count, and write
/tmp/rl_status.json for rl_dashboard.py. Loops every 30s. Gauntlet verdict table is baked in
(those runs are final); the RL curves update live.
"""
import subprocess, re, json, time

SSH = ["ssh", "-i", "/home/ubuntu/.ssh/vastai_id_ed25519", "-p", "30497",
       "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20", "root@ssh8.vast.ai"]

# DECISIVE bench (2026-06-10): persistent bots, 24 g/opp, duplicate walls, stuck 4.6%.
# lad_chunjiandu beat distill100b in 5/6 matchups (+181 net / 144 games) — 3rd independent
# eval favoring it. Older 12-g/opp numbers below the divider are NOT comparable (noisy bench).
GAUNTLET = [
    {"name": "lad_chunjiandu",  "net": 4119, "stuck": 7, "note": "CLEAN 24g/opp — wins 5/6 matchups, +181 over floor"},
    {"name": "distill100b",     "net": 3938, "stuck": 7, "note": "CLEAN 24g/opp — SL floor, current submission"},
    {"name": "— older 12g/opp (noisy bench, not comparable) —", "net": 0, "stuck": 0, "note": ""},
    {"name": "RL league r1",     "net": 1881, "stuck": 8,  "note": "PPO+diverse pool — no gain over its base"},
    {"name": "ens_big_b1.0_s77", "net": 1754, "stuck": 9,  "note": "chun_big (14.7k)"},
    {"name": "ens_union",        "net": 1736, "stuck": 12, "note": "chun+alltop30 union (80k)"},
    {"name": "cl curriculum b03","net": 1687, "stuck": 18, "note": "curriculum RL — below SL"},
    {"name": "ens_top30recent",  "net": 1601, "stuck": 11, "note": "clean recent top-30 (depolluted)"},
    {"name": "ens_top30_b1.0",   "net": 1593, "stuck": 12, "note": "alltop30 (version-polluted)"},
]

LINE = re.compile(r"it (\d+)/(\d+) main_r=([-+0-9.]+) kl=([0-9.]+) beta=([0-9.]+) "
                  r"exp_r=([-+0-9.]+) exp_wr_vs_main=([0-9.]+) m=(\d+) e=(\d+) \((\d+)s\)")


def pull():
    try:
        out = subprocess.run(SSH + [
            "cat /root/mahjong/rl_league.log 2>/dev/null; echo '@@@'; "
            "ls /tmp/leaguepool/m_*.pkl 2>/dev/null | wc -l; "
            "grep -c '^DONE' /root/mahjong/rl_league.log 2>/dev/null; "
            "echo '@@@VAL'; cat /tmp/rl_val.json 2>/dev/null"],
            capture_output=True, text=True, timeout=40).stdout
    except Exception:
        return None
    out, _, valblob = out.partition("@@@VAL")
    val = []
    try:
        val = [{"it": p["it"], "net": p["net"]} for p in json.loads(valblob.strip()).get("points", [])]
    except Exception:
        val = []
    log, _, rest = out.partition("@@@")
    iters, total = [], 500
    for m in LINE.finditer(log):
        total = int(m.group(2))
        iters.append({"it": int(m.group(1)), "main_r": float(m.group(3)), "kl": float(m.group(4)),
                      "beta": float(m.group(5)), "exp_r": float(m.group(6)), "exp_wr": float(m.group(7)),
                      "m": int(m.group(8)), "e": int(m.group(9)), "sec": int(m.group(10))})
    parts = rest.split()
    snaps = int(parts[0]) if parts and parts[0].isdigit() else 0
    done = len(parts) > 1 and parts[1] == "1"
    tail = [l for l in log.splitlines() if l.startswith("it ")]
    return {"updated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "base": "lad_chunjiandu (KL-leashed)", "anchors": 12, "snapshots": snaps,
            "total_iters": total, "iters": iters, "gauntlet": GAUNTLET, "val": val,
            "done": done, "tail": tail[-1] if tail else ""}


if __name__ == "__main__":
    while True:
        st = pull()
        if st:
            json.dump(st, open("/tmp/rl_status.json", "w"))
        time.sleep(30)
