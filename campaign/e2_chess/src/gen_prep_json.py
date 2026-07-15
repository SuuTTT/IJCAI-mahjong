#!/usr/bin/env python3
"""Assemble /root/e2_chess/E2_PREP.json from prep artifacts."""
import glob, json, os, shutil, subprocess, sys, time

ROOT = "/root/e2_chess"


def jload(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def pgn_stats(path):
    if not os.path.exists(path):
        return None
    n = int(subprocess.run(["grep", "-c", "^\\[Event ", path],
                           capture_output=True, text=True).stdout.strip() or 0)
    return {"games": n, "bytes": os.path.getsize(path)}


out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "box": "vast A4000 202.122.49.242:23543", "root": ROOT}

out["data"] = {
    "month_stream": {
        "source": "https://database.lichess.org/standard/lichess_db_standard_rated_2026-06.pgn.zst",
        "filters": "Rated Classical/Rapid, Termination=Normal, both Elo in band, %clk present, reached move 6",
        "status": jload(f"{ROOT}/logs/filter_status.json"),
    },
    "elite_backfill": {
        "source": "https://database.nikonoel.fr/lichess_elite_YYYY-MM.zip (2025-11 backward)",
        "filters": "same, minus %clk (elite db strips clocks); blitz dropped; both Elo >= 2400",
        "status": jload(f"{ROOT}/logs/elite_backfill_status.json"),
    },
    "bands": {os.path.basename(p): pgn_stats(p)
              for p in sorted(glob.glob(f"{ROOT}/data/band_*.pgn"))},
}

enc = {}
for d in sorted(glob.glob(f"{ROOT}/enc/sample_*")):
    s = jload(os.path.join(d, "encode_stats.json"), {})
    s["npz_bytes"] = sum(os.path.getsize(p) for p in glob.glob(d + "/*.npz"))
    enc[os.path.basename(d)] = s
out["encode_samples"] = enc
out["encoding"] = {"obs": "(18,8,8) uint8: 12 piece planes + stm + 4 castling + ep; absolute orientation",
                   "action": "AlphaZero 4672 = from_sq*73 + move_plane; see src/chess_enc.py",
                   "selftest": "enc_selftest.py: 200 games, all legal moves round-trip, no collisions"}

out["ladder_smoke"] = jload(f"{ROOT}/logs/ladder_smoke.json")
out["throughput"] = jload(f"{ROOT}/logs/throughput_probe.json")

du = shutil.disk_usage("/")
out["disk"] = {"total_gb": round(du.total / 2**30, 1), "free_gb": round(du.free / 2**30, 1),
               "e2_chess_gb": round(sum(os.path.getsize(os.path.join(r, f))
                                        for r, _, fs in os.walk(ROOT) for f in fs) / 2**30, 2)}

blockers = []
st = out["data"]["month_stream"]["status"] or {}
for b, n in (st.get("counts") or {}).items():
    if b != "2400plus" and n < 200000:
        blockers.append(f"band {b} below 200k ({n})")
eb = (out["data"]["elite_backfill"]["status"] or {}).get("kept_total", 0)
mo = (st.get("counts") or {}).get("2400plus", 0)
out["band_2400plus_total_games"] = eb + mo
if eb + mo < 200000:
    blockers.append(f"2400plus total {eb + mo} < 200k")
if out["disk"]["free_gb"] < 5:
    blockers.append(f"disk free {out['disk']['free_gb']}GB < 5GB")
for k, s in enc.items():
    if s.get("illegal", 0) > 0 or s.get("games_bad", 0) > 0:
        blockers.append(f"{k}: bad={s.get('games_bad')} illegal={s.get('illegal')}")
out["blockers"] = blockers

with open(f"{ROOT}/E2_PREP.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
