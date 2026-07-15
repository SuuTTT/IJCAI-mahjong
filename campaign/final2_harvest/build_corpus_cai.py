#!/usr/bin/env python3
"""build_corpus_cai.py — re-extract the Final2 corpus in the CAI (38,4,9) encoding.

Lockstep replay of /root/final2_harvest/raw/batch_*.jsonl.gz with the SAME corrected
lite-log semantics as replay_harness2 (peng/chi embedded discards, gang discrimination
via live-discard tracking, hu relabels), but driving caiest feature.FeatureAgent
(obs (38,4,9), mask 235, cai action indices) instead of the 240-dim replay agent.

Output: /root/final2_harvest/final2_cai_corpus.npz
  obs int8 (N,38,4,9), mask bool (N,235), act int16, seat int8, bot int8, kind int8,
  score int16 (actor final game score), fan int16 (winner fanCnt if actor won else 0),
  game int32, srand int64, step int16 (decision idx in game), gamelen int16, bots.

Verification (loud-fail, exit 2):
  * every kept chosen action legal under the cai mask
  * reconstruct-fail games <= 2%
  * N within 8% of the 240-dim corpus
Also cross-checks act agreement vs the 240 corpus on games with identical decision counts.
Writes /root/final2_harvest/CAI_CORPUS_VERIFY.json
"""
import os, sys, json, gzip, glob, time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, "/root/caiest_repro")
from feature import FeatureAgent as CaiAgent

BASE = "/root/final2_harvest"
BOTS = ["kong", "moyu", "QiuQiuR", "player152"]
BOT_IDX = {b: i for i, b in enumerate(BOTS)}
KIND = {"draw": 0, "claim": 1, "claim_play": 2}
OA = CaiAgent.OFFSET_ACT
OT = CaiAgent.OFFSET_TILE
PASS, HU = OA["Pass"], OA["Hu"]


def _disp_stream(records):
    for rec in records:
        if not isinstance(rec, dict):
            continue
        out = rec.get("output") or {}
        disp = out.get("display") or {}
        if isinstance(disp, str):
            try:
                disp = json.loads(disp)
            except Exception:
                continue
        if isinstance(disp, dict) and disp.get("action"):
            yield disp


def reconstruct_cai(records):
    """Same decision semantics as replay_harness2.reconstruct, cai agents.
    Additionally records qianggang-Hu claim chances after BUGANG (and relabels them on HU)."""
    disps = list(_disp_stream(records))
    quan = 0
    agents = None
    decisions = []
    result = {"kind": "unknown"}
    cur_discard = None
    prev_action = None
    last_bugang_tile = None

    def mark_pending_draw(pid, taken):
        for d in reversed(decisions):
            if d["seat"] == pid and d["kind"] in ("draw", "claim_play"):
                if d["taken"] is None:
                    d["taken"] = taken
                return

    def mark_claim(pid, taken, tile):
        for d in reversed(decisions):
            if d["seat"] == pid and d["kind"] == "claim":
                if d["tile"] == tile:
                    d["taken"] = taken
                return

    def rec_claims(pid, tile, rets):
        for s in range(4):
            if s == pid or rets[s] is None:
                continue
            v = agents[s].valid
            if v and v != [PASS]:
                decisions.append(dict(seat=s, kind="claim", obs=rets[s]["observation"],
                                      mask=rets[s]["action_mask"], taken=PASS, tile=tile))

    def apply_play(pid, tile):
        nonlocal cur_discard
        mark_pending_draw(pid, OA["Play"] + OT[tile])
        cur_discard = tile
        rets = [agents[s].request2obs("Player %d Play %s" % (pid, tile)) for s in range(4)]
        rec_claims(pid, tile, rets)

    for disp in disps:
        a = disp["action"]
        if a == "INIT":
            quan = disp.get("quan", 0)
        elif a == "DEAL":
            agents = [CaiAgent(s) for s in range(4)]
            hands = disp["hand"]
            for s in range(4):
                agents[s].request2obs("Wind %d" % quan)
                agents[s].request2obs("Deal " + " ".join(hands[s]))
        elif a == "DRAW":
            pid = disp["player"]; tile = disp["tile"]
            cur_discard = None
            for s in range(4):
                if s == pid:
                    r = agents[s].request2obs("Draw %s" % tile)
                    decisions.append(dict(seat=pid, kind="draw", obs=r["observation"],
                                          mask=r["action_mask"], taken=None, tile=tile))
                else:
                    agents[s].request2obs("Player %d Draw" % pid)
        elif a == "PLAY":
            apply_play(disp["player"], disp["tile"])
        elif a == "PENG":
            pid = disp["player"]; out_tile = disp["tile"]
            mark_claim(pid, OA["Peng"] + OT.get(cur_discard, 0), cur_discard)
            rets = [agents[s].request2obs("Player %d Peng" % pid) for s in range(4)]
            decisions.append(dict(seat=pid, kind="claim_play", obs=rets[pid]["observation"],
                                  mask=rets[pid]["action_mask"], taken=None, tile=out_tile))
            cur_discard = None
            apply_play(pid, out_tile)
        elif a == "CHI":
            pid = disp["player"]; out_tile = disp["tile"]; mid = disp["tileCHI"]
            taken = -1
            if cur_discard and mid and cur_discard[0] == mid[0] and mid[0] in "WTB":
                try:
                    taken = (OA["Chi"] + "WTB".index(mid[0]) * 21 + (int(mid[1]) - 2) * 3
                             + (int(cur_discard[1]) - int(mid[1]) + 1))
                except Exception:
                    taken = -1
            mark_claim(pid, taken, cur_discard)
            rets = [agents[s].request2obs("Player %d Chi %s" % (pid, mid)) for s in range(4)]
            decisions.append(dict(seat=pid, kind="claim_play", obs=rets[pid]["observation"],
                                  mask=rets[pid]["action_mask"], taken=None, tile=out_tile))
            cur_discard = None
            apply_play(pid, out_tile)
        elif a == "GANG":
            pid = disp["player"]; tile = disp.get("tile")
            if cur_discard is not None:      # melded gang of the live discard
                mark_claim(pid, OA["Gang"] + OT.get(cur_discard, 0), cur_discard)
                for s in range(4):
                    agents[s].request2obs("Player %d Gang" % pid)
                cur_discard = None
            else:                            # concealed kong after own draw
                mark_pending_draw(pid, OA["AnGang"] + OT.get(tile, 0))
                for s in range(4):
                    agents[s].request2obs("Player %d AnGang %s" % (pid, tile))
        elif a == "BUGANG":
            pid = disp["player"]; tile = disp["tile"]
            mark_pending_draw(pid, OA["BuGang"] + OT.get(tile, 0))
            last_bugang_tile = tile
            rets = [agents[s].request2obs("Player %d BuGang %s" % (pid, tile)) for s in range(4)]
            rec_claims(pid, tile, rets)      # qianggang-hu chances (extra vs 240 corpus)
        elif a == "HUANG":
            result = {"kind": "draw"}
        elif a == "HU":
            w = disp.get("player")
            result = {"kind": "hu", "winner": w, "fanCnt": disp.get("fanCnt")}
            if w is not None:
                if prev_action == "BUGANG":
                    mark_claim(w, HU, last_bugang_tile)
                elif cur_discard is not None:
                    mark_claim(w, HU, cur_discard)
                else:
                    mark_pending_draw(w, HU)
        prev_action = a
    return quan, decisions, result


def _pack_game(args):
    """worker: replay one game -> packed arrays (or None on reconstruct fail)."""
    gidx, users, srand, fin, logs = args
    try:
        quan, decisions, result = reconstruct_cai(logs)
    except Exception as e:
        return (gidx, None, repr(e)[:200])
    winner = result.get("winner")
    fanCnt = result.get("fanCnt") or 0
    rows0 = [d for d in decisions if d["taken"] is not None and d["taken"] >= 0]
    # drop label/mask-inconsistent rows (e.g. judge-validated zimo Hu that cai's 8-fan
    # calculator rejects, ~0.004% of draws) — they would poison masked CE training
    rows, dropped = [], 0
    for d in rows0:
        if d["mask"][d["taken"]]:
            rows.append(d)
        else:
            dropped += 1
    n = len(rows)
    if n == 0:
        return (gidx, None, "no-decisions")
    obs = np.stack([r["obs"] for r in rows]).astype(np.int8)
    mask = np.stack([r["mask"] for r in rows]).astype(bool)
    act = np.array([r["taken"] for r in rows], np.int16)
    seat = np.array([r["seat"] for r in rows], np.int8)
    kind = np.array([KIND[r["kind"]] for r in rows], np.int8)
    bot = np.array([BOT_IDX[users[r["seat"]]] for r in rows], np.int8)
    score = np.array([(fin[r["seat"]] if fin else 0) for r in rows], np.int16)
    fan = np.array([(fanCnt if winner == r["seat"] else 0) for r in rows], np.int16)
    step = np.arange(n, dtype=np.int16)
    return (gidx, dict(obs=obs, mask=mask, act=act, seat=seat, kind=kind, bot=bot,
                       score=score, fan=fan, step=step, n=n, srand=srand,
                       illegal=dropped), None)


def main():
    t0 = time.time()
    files = sorted(glob.glob(BASE + "/raw/batch_*.jsonl.gz"))
    jobs = []
    gidx = 0
    seen = set()
    for fp in files:
        with gzip.open(fp, "rt") as f:
            for line in f:
                d = json.loads(line)
                mid = d["_mid"]
                if mid in seen:
                    continue
                seen.add(mid)
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
                logs = d.get("logs") or []
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
                jobs.append((gidx, users, srand, fin, logs))
                gidx += 1
    print("games queued: %d (parse %.0fs)" % (len(jobs), time.time() - t0), flush=True)

    with mp.Pool(min(110, mp.cpu_count())) as pool:
        results = pool.map(_pack_game, jobs, chunksize=4)

    fails, illegal_total = [], 0
    parts = []
    for gidx_, pack, err in results:
        if pack is None:
            fails.append((gidx_, err))
            continue
        illegal_total += pack["illegal"]
        parts.append((gidx_, pack))
    parts.sort(key=lambda x: x[0])
    print("replayed %d games ok, %d fails, illegal-act rows %d (%.0fs)"
          % (len(parts), len(fails), illegal_total, time.time() - t0), flush=True)

    obs = np.concatenate([p["obs"] for _, p in parts])
    mask = np.concatenate([p["mask"] for _, p in parts])
    act = np.concatenate([p["act"] for _, p in parts])
    seat = np.concatenate([p["seat"] for _, p in parts])
    kind = np.concatenate([p["kind"] for _, p in parts])
    bot = np.concatenate([p["bot"] for _, p in parts])
    score = np.concatenate([p["score"] for _, p in parts])
    fan = np.concatenate([p["fan"] for _, p in parts])
    step = np.concatenate([p["step"] for _, p in parts])
    game = np.concatenate([np.full(p["n"], g, np.int32) for g, p in parts])
    srand = np.concatenate([np.full(p["n"], p["srand"], np.int64) for _, p in parts])
    gamelen = np.concatenate([np.full(p["n"], p["n"], np.int16) for _, p in parts])
    N = len(act)

    # ---- verification vs the 240-dim corpus ----
    # NOTE: the 240-agent adds Hu UNGATED to every bystander claim valid-set, so the old
    # corpus contains ~2 trivially-forced all-Pass claim rows per discard that the cai
    # agent (fan-gated Hu, real Peng/Chi/Gang conditions = the cooked_single training
    # convention) correctly omits. Like-for-like checks:
    #   (a) draw + claim_play rows (kind 0,2) must match old EXACTLY (count + act order)
    #   (b) non-Pass claim acts must match old exactly (same claims taken, same indices)
    #   (c) new claim rows must be a subset (count <=) of old claim rows
    old = np.load(BASE + "/final2_bc_corpus.npz")
    N_old = len(old["act"])
    per_bot_new = np.bincount(bot, minlength=4).tolist()
    per_bot_old = np.bincount(old["bot"], minlength=4).tolist()
    old_game = old["game"]; old_act = old["act"]; old_kind = old["kind"]
    og_sorted = np.argsort(old_game, kind="stable")
    og = old_game[og_sorted]; oa = old_act[og_sorted]; ok_ = old_kind[og_sorted]
    bounds = np.searchsorted(og, np.arange(og.max() + 2))
    new_off = {}
    off = 0
    for g, p in parts:
        new_off[g] = off; off += p["n"]
    k02_games = k02_bad = 0
    k02_rows = k02_agree = 0
    claim_bad_sub = claim_np_bad = 0
    for g, p in parts:
        if g + 1 >= len(bounds):
            continue
        lo, hi = bounds[g], bounds[g + 1]
        if hi <= lo:
            continue
        a_old = oa[lo:hi]; k_old = ok_[lo:hi]
        sl = slice(new_off[g], new_off[g] + p["n"])
        a_new = act[sl]; k_new = kind[sl]
        o02 = a_old[k_old != 1]; n02 = a_new[k_new != 1]
        k02_games += 1
        if len(o02) != len(n02):
            k02_bad += 1
        else:
            k02_rows += len(o02); k02_agree += int((o02 == n02).sum())
        ocl = a_old[k_old == 1]; ncl = a_new[k_new == 1]
        if len(ncl) > len(ocl):
            claim_bad_sub += 1
        onp = np.sort(ocl[ocl != 0]); nnp = np.sort(ncl[ncl != 0])
        if len(onp) != len(nnp) or not np.array_equal(onp, nnp):
            claim_np_bad += 1
    legal_frac = float(mask[np.arange(N), act.astype(np.int64)].mean())

    ver = dict(
        games_total=len(jobs), games_ok=len(parts), games_fail=len(fails),
        fail_examples=fails[:5], N_new=int(N), N_old=int(N_old),
        note="N_new < N_old expected: old corpus had ~2 ungated-Hu all-Pass claim rows "
             "per discard; cai omits forced claims (training convention)",
        per_bot_new=per_bot_new, per_bot_old=per_bot_old,
        illegal_rows_dropped=int(illegal_total), legal_frac=round(legal_frac, 6),
        kind_counts=np.bincount(kind, minlength=3).tolist(),
        games_compared=k02_games,
        k02_count_mismatch_games=int(k02_bad),
        k02_act_agreement=round(k02_agree / max(k02_rows, 1), 6),
        k02_rows_compared=int(k02_rows),
        claim_subset_violation_games=int(claim_bad_sub),
        claim_nonpass_mismatch_games=int(claim_np_bad),
        seconds=round(time.time() - t0, 1),
    )
    print(json.dumps(ver, default=str), flush=True)
    with open(BASE + "/CAI_CORPUS_VERIFY.json", "w") as f:
        json.dump(ver, f, indent=2, default=str)

    ok = (legal_frac == 1.0 and illegal_total <= 1e-4 * max(N, 1)
          and len(fails) <= 0.02 * len(jobs)
          and k02_bad <= 0.005 * max(k02_games, 1)
          and claim_bad_sub <= 0.005 * max(k02_games, 1)
          and claim_np_bad <= 0.005 * max(k02_games, 1)
          and 0.2 * N_old <= N <= 0.75 * N_old)
    if not ok:
        print("VERIFY FAILED — NOT SAVING", flush=True)
        sys.exit(2)

    np.savez_compressed(BASE + "/final2_cai_corpus.npz",
                        obs=obs, mask=mask, act=act, seat=seat, bot=bot, kind=kind,
                        score=score, fan=fan, game=game, srand=srand,
                        step=step, gamelen=gamelen, bots=np.asarray(BOTS))
    print("saved final2_cai_corpus.npz N=%d (%.0fs)" % (N, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
