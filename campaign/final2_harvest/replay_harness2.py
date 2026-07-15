"""replay_harness2 — corrected lockstep replay for Botzone Chinese-Standard lite logs.

Fixes over IJCAI-mahjong/eval/replay_harness.reconstruct:
  1. PENG/CHI displays embed the claimer's follow-up DISCARD in `tile`
     (tileCHI is the chi's mid tile). Stock code dropped that discard ->
     state drift, missing decision points, ~2.6% hard crashes.
  2. GANG discrimination via live-discard tracking: a melded Gang requires an
     unclaimed discard on the table; GANG with no live discard = AnGang
     (concealed, after own draw; tile revealed in lite logs).
  3. BUGANG displays handled (stock ignored them entirely).
  4. Outcome labels: ron relabels the winner's claim decision (tile-checked)
     to Hu; zimo labels the pending draw decision Hu; qianggang (HU right
     after BUGANG) relabels nothing. AnGang/BuGang label the pending draw
     decision. Stock left ron mislabeled as Pass.

Decision kinds: 'draw' (after own draw), 'claim' (react to a discard),
'claim_play' (discard after chi/peng).
"""
import json
import sys
import numpy as np
sys.path.insert(0, "/root/IJCAI-mahjong-full/IJCAI-mahjong")
from data.feature_agent import (FeatureAgent, ACT, ACT_DIM, TILE_INDEX,
                                chi_action_idx)


def _mask(agent):
    m = np.zeros(ACT_DIM, dtype=bool)
    for v in agent.valid:
        if 0 <= v < ACT_DIM:
            m[v] = True
    return m


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


def reconstruct(records):
    """records: the match's `logs` list. Returns (quan, decisions, result)."""
    disps = list(_disp_stream(records))
    quan = 0
    agents = None
    decisions = []
    result = {"kind": "unknown"}
    cur_discard = None      # tile of the live (unclaimed) discard, else None
    prev_action = None

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

    def snapshot_claims(pid):
        for s in range(4):
            if s != pid and agents[s].valid and agents[s].valid != [ACT["Pass"]]:
                decisions.append(dict(idx=len(decisions), seat=s, kind="claim",
                                      obs=agents[s].obs.copy(), mask=_mask(agents[s]),
                                      taken=ACT["Pass"], tile=cur_discard))

    def apply_play(pid, tile, from_draw):
        nonlocal cur_discard
        if from_draw:
            mark_pending_draw(pid, ACT["Play"] + TILE_INDEX[tile])
        cur_discard = tile
        msg = f"Player {pid} Play {tile}"
        for s in range(4):
            agents[s].update(msg)
        snapshot_claims(pid)

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
            cur_discard = None   # unclaimed discard is dead once someone draws
            for s in range(4):
                agents[s].update(f"Draw {tile}" if s == pid else f"Player {pid} Draw")
            decisions.append(dict(idx=len(decisions), seat=pid, kind="draw",
                                  obs=agents[pid].obs.copy(), mask=_mask(agents[pid]),
                                  taken=None, tile=tile))
        elif a == "PLAY":
            apply_play(disp["player"], disp["tile"], from_draw=True)
        elif a == "PENG":
            pid = disp["player"]; out_tile = disp["tile"]
            mark_claim(pid, ACT["Peng"] + TILE_INDEX.get(cur_discard, 0), cur_discard)
            for s in range(4):
                agents[s].update(f"Player {pid} Peng")
            cur_discard = None
            decisions.append(dict(idx=len(decisions), seat=pid, kind="claim_play",
                                  obs=agents[pid].obs.copy(), mask=_mask(agents[pid]),
                                  taken=None, tile=out_tile))
            apply_play(pid, out_tile, from_draw=True)  # labels the claim_play
        elif a == "CHI":
            pid = disp["player"]; out_tile = disp["tile"]; mid = disp["tileCHI"]
            taken = -1
            if cur_discard and mid and cur_discard[0] == mid[0]:
                try:
                    taken = chi_action_idx(mid[0], int(mid[1]), int(cur_discard[1]))
                except Exception:
                    taken = -1
            mark_claim(pid, taken, cur_discard)
            for s in range(4):
                agents[s].update(f"Player {pid} Chi {mid}")
            cur_discard = None
            decisions.append(dict(idx=len(decisions), seat=pid, kind="claim_play",
                                  obs=agents[pid].obs.copy(), mask=_mask(agents[pid]),
                                  taken=None, tile=out_tile))
            apply_play(pid, out_tile, from_draw=True)
        elif a == "GANG":
            pid = disp["player"]; tile = disp.get("tile")
            if cur_discard is not None:
                # melded gang of the live discard
                mark_claim(pid, ACT["Gang"] + TILE_INDEX.get(cur_discard, 0), cur_discard)
                for s in range(4):
                    agents[s].update(f"Player {pid} Gang")
                cur_discard = None
            else:
                # concealed kong after own draw
                mark_pending_draw(pid, ACT["AnGang"] + TILE_INDEX.get(tile, 34))
                for s in range(4):
                    agents[s].update(f"Player {pid} AnGang {tile}")
            # replacement draw arrives as the next DRAW display
        elif a == "BUGANG":
            pid = disp["player"]; tile = disp["tile"]
            mark_pending_draw(pid, ACT["BuGang"] + TILE_INDEX.get(tile, 0))
            for s in range(4):
                agents[s].update(f"Player {pid} BuGang {tile}")
        elif a == "HUANG":
            result = {"kind": "draw"}
        elif a == "HU":
            w = disp.get("player")
            result = {"kind": "hu", "winner": w,
                      "fanCnt": disp.get("fanCnt"), "score": disp.get("score")}
            if w is not None:
                if prev_action == "BUGANG":
                    pass  # qianggang: no snapshot to relabel
                elif cur_discard is not None:
                    mark_claim(w, ACT["Hu"], cur_discard)
                else:
                    mark_pending_draw(w, ACT["Hu"])
        prev_action = a
    return quan, decisions, result
