#!/usr/bin/env python3
"""Build final2_bc_corpus.npz: ALL 4 bots' decisions from Final2 logs via
FeatureAgent lockstep replay (corrected harness: replay_harness2)."""
import os, sys, json, gzip, glob
sys.path.insert(0, "/root/final2_harvest")
import numpy as np
from replay_harness2 import reconstruct

BASE = "/root/final2_harvest"
BOTS = ["kong", "moyu", "QiuQiuR", "player152"]
BOT_IDX = {b: i for i, b in enumerate(BOTS)}
KIND = {"draw": 0, "claim": 1, "claim_play": 2}

obs_l, mask_l, act_l, seat_l, bot_l, kind_l, score_l, fan_l, gidx_l, srand_l = \
    [], [], [], [], [], [], [], [], [], []

files = sorted(glob.glob(f"{BASE}/raw/batch_*.jsonl.gz"))
gidx = 0
fails = 0
seen = set()

for fp in files:
    with gzip.open(fp, "rt") as f:
        for line in f:
            d = json.loads(line)
            mid = d["_mid"]
            if mid in seen:
                continue
            seen.add(mid)
            logs = d.get("logs") or []
            users = []
            for p in d.get("players", []):
                nm = p.get("name", "")
                users.append(nm.split("]")[0].lstrip("[") if "]" in nm else nm)
            if len(users) != 4 or any(u not in BOT_IDX for u in users):
                continue
            try:
                init = json.loads(d.get("initdata") or "{}")
            except Exception:
                init = {}
            srand = init.get("srand") or 0

            fin = None
            for rec in reversed(logs):
                if isinstance(rec, dict) and "output" in rec:
                    out = rec["output"] or {}
                    if out.get("command") == "finish":
                        c = out.get("content") or {}
                        try:
                            fin = [int(c[str(i)]) for i in range(4)]
                        except Exception:
                            fin = None
                        break

            try:
                quan, decisions, result = reconstruct(logs)
            except Exception:
                fails += 1
                gidx += 1
                continue

            winner = result.get("winner")
            fanCnt = result.get("fanCnt") or 0
            for dd in decisions:
                t = dd["taken"]
                if t is None or t < 0:
                    continue
                s = dd["seat"]
                obs_l.append(np.asarray(dd["obs"], np.uint8))
                mask_l.append(np.asarray(dd["mask"], bool))
                act_l.append(t)
                seat_l.append(s)
                bot_l.append(BOT_IDX[users[s]])
                kind_l.append(KIND[dd["kind"]])
                score_l.append(fin[s] if fin else 0)
                fan_l.append(fanCnt if winner == s else 0)
                gidx_l.append(gidx)
                srand_l.append(srand)
            gidx += 1
            if gidx % 1000 == 0:
                print(f"{gidx} games, {len(act_l)} samples, {fails} fails", flush=True)

print(f"TOTAL {gidx} games, {len(act_l)} samples, {fails} reconstruct-fails")
np.savez_compressed(
    f"{BASE}/final2_bc_corpus.npz",
    obs=np.stack(obs_l).astype(np.uint8),
    mask=np.stack(mask_l),
    act=np.asarray(act_l, np.int16),
    seat=np.asarray(seat_l, np.int8),
    bot=np.asarray(bot_l, np.int8),
    kind=np.asarray(kind_l, np.int8),
    score=np.asarray(score_l, np.int16),
    fan=np.asarray(fan_l, np.int16),
    game=np.asarray(gidx_l, np.int32),
    srand=np.asarray(srand_l, np.int64),
    bots=np.asarray(BOTS),
)
print("saved final2_bc_corpus.npz")
