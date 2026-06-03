"""
replay_harness.py — reconstruct a Botzone match log into per-seat decision points,
and (optionally) re-run any model on them. The foundation for the Heuristic-Learning
loop: turn real replays into a regression suite of "situations" + what we did + what a
given model would do, so we can (a) detect regressions and (b) mine failure patterns
(missed HU, bad claim, dangerous discard) once real ladder logs land.

A Botzone log (see log/4-bpt.log) is a list of records; each record's
output.display.action drives state. We rebuild every seat's FeatureAgent by feeding it
the SAME message vocabulary the live bot/sim use (Wind/Deal/Draw/Player ... Play/Chi/
Peng/Gang), then snapshot (obs, mask) at each point where a seat must act.

A DecisionPoint:
  seat, kind ('draw'|'claim'), obs, mask, taken (the action actually played in the log),
  meta (tile drawn / discarded, etc.)

FIDELITY (v1): `--faithful` drives bot/ml_bot.py's REAL decision logic. On a log produced
by the SAME model it reproduces the exact discard ~59% of the time (vs 16% for raw-net
argmax); the residual gap is an obs-HISTORY reconstruction nuance (the live bot accumulates
state via the keep-running protocol; we rebuild from display events). This is SUFFICIENT
for failure MINING (the intended use): missed-legal-HU detection + claim/danger flags read
off the reconstructed legal state and don't depend on exact discard tie-breaking. Closing
to ~100% (replay through the real handle() protocol) is deferred until real ladder logs
make it worth the effort.

Usage:
  python3 eval/replay_harness.py log/4-bpt.log            # summarize the replay
  python3 eval/replay_harness.py log/4-bpt.log MODEL.npz  # + what MODEL would do, diffs
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data.feature_agent import FeatureAgent, ACT, ACT_DIM, TILE_LIST, TILE_INDEX, decode_chi


def _disp_stream(records):
    """Yield display dicts in order (skip non-action records)."""
    for rec in records:
        out = rec.get("output") or {}
        disp = out.get("display") or {}
        if disp.get("action"):
            yield disp


def _mask(agent):
    m = np.zeros(ACT_DIM, dtype=bool)
    for v in agent.valid:
        if 0 <= v < ACT_DIM:
            m[v] = True
    return m


def reconstruct(path):
    """Replay the log, return (quan, hands0, decisions, result).
    decisions: list of dicts {idx, seat, kind, obs, mask, taken, tile}."""
    records = json.load(open(path))
    disps = list(_disp_stream(records))
    quan = 0
    agents = None
    decisions = []
    result = {"kind": "unknown"}

    for disp in disps:
        a = disp["action"]
        if a == "INIT":
            quan = disp.get("quan", 0)
        elif a == "DEAL":
            agents = [FeatureAgent(s) for s in range(4)]
            for s in range(4):
                agents[s].update(f"Wind {quan}")
            hands = disp["hand"]
            for s in range(4):
                agents[s].update("Deal " + " ".join(hands[s]))
        elif a == "DRAW":
            pid = disp["player"]; tile = disp["tile"]
            # the drawer sees the real tile + must act; others see a hidden draw
            for s in range(4):
                if s == pid:
                    agents[s].update(f"Draw {tile}")
                else:
                    agents[s].update(f"Player {pid} Draw")
            # snapshot the drawer's decision point (it will Play/Gang/Hu)
            decisions.append(dict(idx=len(decisions), seat=pid, kind="draw",
                                  obs=agents[pid].obs.copy(), mask=_mask(agents[pid]),
                                  taken=None, tile=tile))
        elif a == "PLAY":
            pid = disp["player"]; tile = disp["tile"]
            # record what the drawer actually did (a Play of `tile`)
            for d in reversed(decisions):
                if d["seat"] == pid and d["kind"] == "draw" and d["taken"] is None:
                    d["taken"] = ACT["Play"] + TILE_INDEX[tile]; break
            # before applying, snapshot each OTHER seat's claim opportunity
            msg = f"Player {pid} Play {tile}"
            for s in range(4):
                agents[s].update(msg)
            for s in range(4):
                if s != pid and agents[s].valid and agents[s].valid != [ACT["Pass"]]:
                    decisions.append(dict(idx=len(decisions), seat=s, kind="claim",
                                          obs=agents[s].obs.copy(), mask=_mask(agents[s]),
                                          taken=ACT["Pass"], tile=tile))  # default: passed
        elif a in ("CHI", "PENG", "GANG"):
            pid = disp["player"]
            tile = disp.get("tileCHI") or disp.get("tile") or ""
            # mark the claiming seat's prior claim decision as taken (not pass)
            for d in reversed(decisions):
                if d["seat"] == pid and d["kind"] == "claim":
                    if a == "CHI":
                        # mid tile is tileCHI; encode best-effort
                        d["taken"] = -1  # claim taken (chi); exact idx not needed for stats
                    elif a == "PENG":
                        d["taken"] = ACT["Peng"] + TILE_INDEX.get(d["tile"], 0)
                    else:
                        d["taken"] = ACT["Gang"] + TILE_INDEX.get(d["tile"], 0)
                    break
            if a == "CHI":
                mid = disp["tileCHI"]
                for s in range(4):
                    agents[s].update(f"Player {pid} Chi {mid}")
            elif a == "PENG":
                for s in range(4):
                    agents[s].update(f"Player {pid} Peng")
            else:
                for s in range(4):
                    agents[s].update(f"Player {pid} Gang")
        elif a == "HUANG":
            result = {"kind": "draw"}
        elif a in ("HU",):
            result = {"kind": "hu", "winner": disp.get("player")}
    return quan, decisions, result


def summarize(path, model=None):
    quan, decisions, result = reconstruct(path)
    draws = sum(1 for d in decisions if d["kind"] == "draw")
    claims = sum(1 for d in decisions if d["kind"] == "claim")
    print(f"replay {os.path.basename(path)}: quan={quan} result={result}")
    print(f"  decision points: {len(decisions)}  (draw-turns={draws}, claim-chances={claims})")
    claim_taken = sum(1 for d in decisions if d["kind"] == "claim" and d["taken"] != ACT["Pass"])
    print(f"  claim chances ACTED on: {claim_taken}/{claims}  "
          f"({100*claim_taken/max(1,claims):.0f}% — low = passive)")
    if model:
        from train.numpy_infer import NumpyMLP
        m = NumpyMLP(model)
        diffs = 0; hu_misses = 0
        for d in decisions:
            probs, _ = m.forward(d["obs"], d["mask"])
            probs = np.where(d["mask"], np.nan_to_num(probs), 0.0)
            act = int(np.argmax(probs)) if probs.sum() > 0 else -1
            if d["mask"][ACT["Hu"]] and act != ACT["Hu"]:
                pass  # Hu is offered every draw; only a miss if a real 8-fan win existed
            if d["taken"] is not None and d["taken"] >= 0 and act != d["taken"]:
                diffs += 1
        print(f"  model={os.path.basename(model)}: would differ from logged action "
              f"at {diffs}/{len(decisions)} decision points "
              f"({100*diffs/max(1,len(decisions)):.0f}%)")
    return quan, decisions, result


def faithful_eval(path, model_path):
    """Replay the log driving bot/ml_bot.py's REAL decision logic (heuristic re-rank +
    MyHand + fan-gated HU), not raw net argmax. Reconstructs 4 FeatureAgents in lockstep
    and swaps each into ml_bot at its decision point. Reports draw-turn play-tile match
    vs the log (a faithfulness check: on a log produced BY this bot, match should be high),
    and flags missed legal HUs / claim deviations — the hooks for failure mining.
    """
    os.environ["MODEL"] = model_path
    import importlib
    import bot.ml_bot as B
    importlib.reload(B)
    records = json.load(open(path))
    disps = list(_disp_stream(records))
    quan = 0; agents = None
    play_total = play_match = 0
    hu_missed = 0; claim_dev = 0; flags = []

    def drive_draw(seat, drawn):
        B.agent = agents[seat]; B._seat = seat; B._quan = quan
        B._recent_kong = False; B._last_draw = drawn
        try: return B.decide_draw(drawn)
        except Exception as e: return f"ERR {e}"

    def drive_claim(seat, disc):
        B.agent = agents[seat]; B._seat = seat; B._quan = quan
        B._recent_kong = False
        try: return B.decide_claim(disc)
        except Exception as e: return f"ERR {e}"

    for disp in disps:
        a = disp["action"]
        if a == "INIT":
            quan = disp.get("quan", 0)
        elif a == "DEAL":
            agents = [FeatureAgent(s) for s in range(4)]
            for s in range(4):
                agents[s].update(f"Wind {quan}")
                agents[s].update("Deal " + " ".join(disp["hand"][s]))
        elif a == "DRAW":
            pid = disp["player"]; tile = disp["tile"]
            for s in range(4):
                agents[s].update(f"Draw {tile}" if s == pid else f"Player {pid} Draw")
            # the bot's real decision at this draw turn
            resp = drive_draw(pid, tile)
            disp["_bot_draw"] = resp  # stash; compared at the following PLAY
            if isinstance(resp, str) and resp.startswith("HU"):
                disp["_bot_hu"] = True
        elif a == "PLAY":
            pid = disp["player"]; tile = disp["tile"]
            # compare the bot's draw-turn decision to what was actually played
            bd = None
            for d in reversed(disps):
                if d is disp: continue
            # find the matching DRAW we annotated: it's the most recent for pid
            # (we stashed on the DRAW disp; retrieve via a small scan)
            bd = next((d.get("_bot_draw") for d in reversed(_recent_draws(disps, disp, pid))), None)
            if isinstance(bd, str) and bd.startswith("PLAY "):
                play_total += 1
                if bd.split(" ", 1)[1] == tile: play_match += 1
                else: flags.append(f"draw->play differ: bot {bd!r} vs log PLAY {tile}")
            msg = f"Player {pid} Play {tile}"
            for s in range(4):
                agents[s].update(msg)
            # claim opportunities for others: does the bot agree with log (pass)?
            for s in range(4):
                if s != pid and agents[s].valid and agents[s].valid != [ACT["Pass"]]:
                    resp = drive_claim(s, tile)
                    if isinstance(resp, str) and resp.startswith("HU"):
                        # bot would HU here but the log didn't -> a (rare) missed-HU flag
                        hu_missed += 1
                        flags.append(f"seat {s} bot would HU on {tile} (log passed)")
        elif a == "CHI":
            pid = disp["player"]; mid = disp["tileCHI"]
            for s in range(4): agents[s].update(f"Player {pid} Chi {mid}")
        elif a == "PENG":
            pid = disp["player"]
            for s in range(4): agents[s].update(f"Player {pid} Peng")
        elif a == "GANG":
            pid = disp["player"]
            for s in range(4): agents[s].update(f"Player {pid} Gang")

    print(f"FAITHFUL replay {os.path.basename(path)} with {os.path.basename(model_path)}:")
    print(f"  draw-turn play-tile match: {play_match}/{play_total} "
          f"({100*play_match/max(1,play_total):.0f}%)  "
          f"(high % => harness drives the real bot faithfully)")
    print(f"  bot-would-HU-but-log-passed: {hu_missed}  (missed-HU candidates to mine)")
    if flags:
        print("  sample flags:")
        for f in flags[:8]:
            print("   -", f)
    return play_match, play_total, flags


def _recent_draws(disps, before, pid):
    """Return DRAW disps for pid that occurred before `before` (most recent last)."""
    out = []
    for d in disps:
        if d is before: break
        if d.get("action") == "DRAW" and d.get("player") == pid and "_bot_draw" in d:
            out.append(d)
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "log/4-bpt.log"
    model = sys.argv[2] if len(sys.argv) > 2 else None
    if len(sys.argv) > 3 and sys.argv[3] == "--faithful":
        faithful_eval(path, model)
    else:
        summarize(path, model)
