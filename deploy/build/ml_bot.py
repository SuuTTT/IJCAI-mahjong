"""
ml_bot.py — feat_agent-driven hybrid bot (Keep Running + one-shot JSON).

Single source of truth: FeatureAgent (faithful port of the official Botzone
engine). It tracks hand/melds and produces feat.valid = the exact set of LEGAL
actions each turn. Decisions:

  • HU       : only when ACT['Hu'] is legal AND feat.can_hu(...) >= 8 fan
               (independent fan-calculator check — never a wrong-hu / -30).
  • discard  : ML model picks among legal PLAY actions (offense + defense);
               falls back to a shanten heuristic if no model.
  • PENG/CHI/GANG/BUGANG : taken when legal and they reduce shanten
               (model-scored), else PASS. Always chosen from feat.valid, so
               they can never be illegal.

Because every emitted action is drawn from feat.valid (the legal set) and HU is
fan-gated, the bot cannot produce an illegal move (no WA / WH / -30).

Env: MODEL (npz weights), ML_DEBUG (log file path).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from feature_agent import (
    FeatureAgent, ACT, ACT_DIM, TILE_LIST, TILE_INDEX, decode_chi, HAS_FAN,
)

# pure-python shanten fallback for discard scoring when no model
from mahjong_bot import shanten as _shanten

SENTINEL   = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"
MODEL_PATH = os.environ.get("MODEL", "train/checkpoints/bc_v3_ft_weights.npz")
DEBUG      = os.environ.get("ML_DEBUG", "")
_dbg = open(DEBUG, "a") if DEBUG else None

model = None
_model_err = ""
if os.path.exists(MODEL_PATH):
    try:
        from numpy_infer import NumpyMLP
        model = NumpyMLP(MODEL_PATH)
    except Exception as e:
        _model_err = f"load_fail:{e}"
        if _dbg: _dbg.write(f"model load failed: {e}\n")
else:
    _model_err = f"missing:{MODEL_PATH}"


def env_status():
    """One-line environment report, surfaced via the JSON `debug` field so it
    shows up in Botzone's Debug Mode log. Tells us if MahjongGB (needed to HU)
    and the model are actually available on the server."""
    return (f"mahjong={'OK' if HAS_FAN else 'MISSING'} "
            f"model={'OK' if model is not None else 'NONE('+_model_err+')'} "
            f"path={os.path.basename(MODEL_PATH)}")

agent = None          # FeatureAgent
_seat = 0
_quan = 0
_last_draw = None     # tile I just drew (for self-draw HU / wall-last flags)
_last_discard = None  # last tile discarded by an opponent (for rong / claims)


def reset(seat, wind):
    global agent, _seat, _quan, _last_draw, _last_discard
    _seat, _quan = seat, wind
    _last_draw = _last_discard = None
    agent = FeatureAgent(seat)
    agent.update(f"Wind {wind}")


def dbg(req, resp):
    if _dbg:
        _dbg.write(f"req={req!r} resp={resp!r} hand={sorted(agent.hand)} "
                   f"packs={agent.packs[_seat]}\n"); _dbg.flush()


# ── model scoring over a legal action subset ──────────────────────────────────

def _model_probs(valid):
    mask = np.zeros(ACT_DIM, dtype=bool)
    for v in valid:
        if 0 <= v < ACT_DIM:
            mask[v] = True
    probs, _ = model.forward(agent.obs, mask)
    return probs


def _best_play_tile(valid):
    """Pick discard tile from legal PLAY actions (model if available)."""
    plays = [v for v in valid if ACT["Play"] <= v < ACT["Chi"]]
    if not plays:
        return None
    if model is not None:
        probs = _model_probs(plays)
        best = max(plays, key=lambda v: probs[v])
        return TILE_LIST[best - ACT["Play"]]
    # heuristic fallback: minimise shanten
    best_t, best_s = None, 99
    for v in plays:
        t = TILE_LIST[v - ACT["Play"]]
        rem = list(agent.hand); rem.remove(t)
        s, _ = _shanten(rem, agent.packs[_seat])
        if s < best_s:
            best_s, best_t = s, t
    return best_t


def _shanten_now():
    s, _ = _shanten(list(agent.hand), agent.packs[_seat])
    return s


def _wall_exhausted():
    """True if any player's wall is empty. The judge forbids CHI/PENG/GANG
    (which need a continuation/replacement draw) once the relevant wall is
    empty — it checks individual walls, not the total. Guarding on the minimum
    wall count matches that and prevents end-of-wall claim/kong -30 penalties."""
    try:
        return min(agent.wall_counts) <= 0
    except Exception:
        return False


# ── decision after my draw (rtype 2) ──────────────────────────────────────────

def decide_draw(drawn):
    valid = agent.valid
    # 1) HU — only if legal AND fan>=8 (fan-gated, judge-consistent)
    if ACT["Hu"] in valid:
        fan = agent.can_hu(drawn, is_self=True,
                           is_kong=(_recent_kong))
        if _dbg: _dbg.write(f"  HU-draw check: win={drawn} fan={fan} "
                            f"hand={sorted(agent.hand)} packs={agent.packs[_seat]} "
                            f"wall_last={agent.wall_last} my_wall_last={agent.my_wall_last} "
                            f"recent_kong={_recent_kong}\n"); _dbg.flush()
        if fan >= 8:
            return "HU"
    s_before = _shanten_now()
    # 2) AnGang / BuGang — skip entirely if wall nearly empty (needs replacement draw)
    if not _wall_exhausted():
        angang = [v for v in valid if ACT["AnGang"] <= v < ACT["BuGang"]]
        for v in angang:
            t = TILE_LIST[v - ACT["AnGang"]]
            rem = [x for x in agent.hand if x != t]
            s, _ = _shanten(rem, agent.packs[_seat] + [("GANG", t, _seat)])
            if s <= s_before:
                return f"GANG {t}"
        bugang = [v for v in valid if v >= ACT["BuGang"]]
        for v in bugang:
            t = TILE_LIST[v - ACT["BuGang"]]
            rem = list(agent.hand); rem.remove(t)
            s, _ = _shanten(rem, agent.packs[_seat])
            if s <= s_before:
                return f"BUGANG {t}"
    # 3) discard
    t = _best_play_tile(valid)
    return f"PLAY {t}" if t else "PASS"


# ── decision after opponent discard (rtype 3 PLAY) ────────────────────────────

def _chi_discard_after(disc_tile, choice):
    """After a model-chosen CHI action index, return 'CHI mid disc' or 'PASS'."""
    suit, mid_n, _ = decode_chi(choice)
    mid_tile = f"{suit}{mid_n}"
    rem = list(agent.hand)
    for d in (-1, 0, 1):
        tt = f"{suit}{mid_n+d}"
        if tt == disc_tile: continue
        if tt in rem: rem.remove(tt)
        else: return "PASS"          # can't actually form it
    disc_after = _peng_discard(rem, agent.packs[_seat] + [("CHI", mid_tile, 1)])
    return f"CHI {mid_tile} {disc_after}" if disc_after else "PASS"


def _peng_response(disc_tile):
    """Return 'PENG disc' after a peng of disc_tile, or 'PASS'."""
    rem, c = [], 0
    for x in agent.hand:
        if x == disc_tile and c < 2: c += 1
        else: rem.append(x)
    disc_after = _peng_discard(rem, agent.packs[_seat] + [("PENG", disc_tile, 0)])
    return f"PENG {disc_after}" if disc_after else "PASS"


def _heuristic_claim(disc_tile, valid):
    """Fallback when no model: claim only if it strictly lowers shanten (the old
    fan-blind rule). Kept so the bot still runs without trained weights."""
    s_before = _shanten_now()
    gang = [v for v in valid if ACT["Gang"] <= v < ACT["AnGang"]]
    if gang:
        t = TILE_LIST[gang[0] - ACT["Gang"]]
        rem = [x for x in agent.hand if x != t]
        s, _ = _shanten(rem, agent.packs[_seat] + [("GANG", t, 0)])
        if s <= s_before: return "GANG"
    peng = [v for v in valid if ACT["Peng"] <= v < ACT["Gang"]]
    if peng:
        rem, c = [], 0
        for x in agent.hand:
            if x == disc_tile and c < 2: c += 1
            else: rem.append(x)
        np_ = agent.packs[_seat] + [("PENG", disc_tile, 0)]
        best = min((_shanten([y for y in rem if y != x], np_)[0] for x in set(rem)), default=99)
        if best < s_before: return _peng_response(disc_tile)
    for v in [v for v in valid if ACT["Chi"] <= v < ACT["Peng"]]:
        r = _chi_discard_after(disc_tile, v)
        if r != "PASS":
            suit, mid_n, _ = decode_chi(v)
            rem = list(agent.hand)
            for d in (-1,0,1):
                tt=f"{suit}{mid_n+d}"
                if tt!=disc_tile and tt in rem: rem.remove(tt)
            best = min((_shanten([y for y in rem if y != x], agent.packs[_seat]+[("CHI",f"{suit}{mid_n}",1)])[0]
                        for x in set(rem)), default=99)
            if best < s_before: return r
    return "PASS"


def decide_claim(disc_tile):
    valid = agent.valid
    # 1) rong HU — highest priority, fan-gated (never trust the model for legality)
    if ACT["Hu"] in valid:
        fan = agent.can_hu(disc_tile, is_self=False)
        if _dbg: _dbg.write(f"  HU-rong check: win={disc_tile} fan={fan}\n"); _dbg.flush()
        if fan >= 8:
            return "HU"
    # No claims once the wall is exhausted (no continuation draw) — only HU above.
    if _wall_exhausted():
        return "PASS"

    # 2) MODEL-DRIVEN claim selection over the legal {Pass, Peng, Gang, Chi} set.
    #    The model learned from strong players WHICH tiles to claim and what to
    #    build toward (碰碰和 / 清一色 / 混一色 …) — i.e. it accounts for fan value,
    #    unlike the old shanten-only rule. Pass is included so it can decline a
    #    tempting claim to chase a higher-value hand.
    claim_acts = [v for v in valid
                  if v == ACT["Pass"] or (ACT["Chi"] <= v < ACT["AnGang"])]
    if model is not None and len(claim_acts) > 1:
        mask = np.zeros(ACT_DIM, dtype=bool)
        for v in claim_acts:
            mask[v] = True
        probs, _ = model.forward(agent.obs, mask)
        choice = max(claim_acts, key=lambda v: probs[v])
        if _dbg:
            _dbg.write(f"  claim choice={choice} from {claim_acts} "
                       f"p={probs[choice]:.2f}\n"); _dbg.flush()
        if choice == ACT["Pass"]:
            return "PASS"
        if ACT["Chi"] <= choice < ACT["Peng"]:
            return _chi_discard_after(disc_tile, choice)
        if ACT["Peng"] <= choice < ACT["Gang"]:
            return _peng_response(disc_tile)
        if ACT["Gang"] <= choice < ACT["AnGang"]:
            return "GANG"
        return "PASS"

    # 3) Fallback (no model): old shanten heuristic.
    return _heuristic_claim(disc_tile, valid)


def _peng_discard(remaining, new_packs):
    """Pick the discard after a peng/chi (model-scored, in `remaining`)."""
    if not remaining:
        return None
    best_t, best_s = None, 99
    for x in set(remaining):
        r = list(remaining); r.remove(x)
        s, _ = _shanten(r, new_packs)
        if s < best_s:
            best_s, best_t = s, x
    return best_t


# ── BUGANG rob-kong (rtype 3 BUGANG) ──────────────────────────────────────────

def decide_robkong(tile):
    valid = agent.valid
    if ACT["Hu"] in valid:
        fan = agent.can_hu(tile, is_self=False, is_kong=True)
        if fan >= 8:
            return "HU"
    return "PASS"


# ── universal emit-time legality verifier (last line of defence vs -30) ───────

def verify_draw(resp, drawn):
    """Validate a response to MY draw. hand currently includes `drawn`.
       Returns resp if physically legal, else a guaranteed-legal PLAY."""
    hand = agent.hand
    parts = resp.split()
    op = parts[0] if parts else ""
    if op == "HU":
        return resp                                  # already fan-gated
    if op == "PLAY" and len(parts) == 2 and parts[1] in hand:
        return resp
    if op == "GANG" and len(parts) == 2 and hand.count(parts[1]) >= 4 \
            and not _wall_exhausted():
        return resp
    if op == "BUGANG" and len(parts) == 2 and parts[1] in hand \
            and any(p[0] == "PENG" and p[1] == parts[1] for p in agent.packs[_seat]) \
            and not _wall_exhausted():
        return resp
    # fallback: discard any tile actually in hand (never illegal)
    return f"PLAY {hand[0]}" if hand else "PASS"


def verify_claim(resp, disc):
    """Validate a claim response to an opponent discard. Hand does NOT include
       `disc`. Returns resp if physically legal, else PASS."""
    hand = agent.hand
    parts = resp.split()
    op = parts[0] if parts else "PASS"
    if op == "PASS":
        return "PASS"
    if op == "HU":
        return resp                                  # fan-gated
    if _wall_exhausted():
        return "PASS"                                # no claims at wall end
    if op == "GANG":
        return "GANG" if hand.count(disc) >= 3 else "PASS"
    if op == "PENG" and len(parts) == 2:
        after = parts[1]
        if hand.count(disc) >= 2:
            # hand after removing 2 disc must still contain the discard tile
            tmp = list(hand); tmp.remove(disc); tmp.remove(disc)
            if after in tmp:
                return resp
        return "PASS"
    if op == "CHI" and len(parts) == 3:
        mid, after = parts[1], parts[2]
        if not mid or mid[0] not in "WTB" or mid[0] != disc[0]:
            return "PASS"
        n = int(mid[1])
        need = [f"{mid[0]}{k}" for k in (n - 1, n, n + 1) if f"{mid[0]}{k}" != disc]
        tmp = list(hand)
        for t in need:
            if t in tmp: tmp.remove(t)
            else: return "PASS"
        if after in tmp:
            return resp
        return "PASS"
    return "PASS"


# ── protocol handler ──────────────────────────────────────────────────────────

_recent_kong = False   # I declared a kong last draw (杠上开花 flag)

def respond(r):
    print(r, flush=True)
    print(SENTINEL, flush=True)


def feat(line):
    """Feed raw request to feat_agent (tracks all players)."""
    parts = line.split(); rt = parts[0]
    if rt == "1":
        agent.update("Deal " + " ".join(parts[5:]))
    elif rt == "2":
        agent.update(f"Draw {parts[1]}")
    elif rt == "3":
        pid = parts[1]; act = parts[2]; rest = parts[3:]
        m = {"DRAW": f"Player {pid} Draw",
             "PLAY": f"Player {pid} Play {rest[0] if rest else ''}",
             "PENG": f"Player {pid} Peng",
             "CHI":  f"Player {pid} Chi {rest[0] if rest else ''}",
             "GANG": f"Player {pid} Gang",
             "BUGANG": f"Player {pid} BuGang {rest[0] if rest else ''}"}
        if act in m:
            agent.update(m[act])


def handle(line):
    global _last_draw, _last_discard, _recent_kong
    line = line.strip()
    parts = line.split()
    if not parts:
        respond("PASS"); return
    rt = parts[0]

    if rt == "0":
        reset(int(parts[1]), int(parts[2])); respond("PASS")

    elif rt == "1":
        feat(line); respond("PASS")

    elif rt == "2":
        _last_draw = parts[1]
        feat(line)                      # adds drawn tile to hand, sets valid
        resp = decide_draw(_last_draw)
        resp = verify_draw(resp, _last_draw)     # universal legality guard
        _recent_kong = resp.startswith("GANG") or resp.startswith("BUGANG")
        # Apply my action to feat NOW (single application point; echo is skipped)
        rp = resp.split()
        if rp[0] == "PLAY":
            agent.update(f"Player {_seat} Play {rp[1]}")
        elif rp[0] == "GANG":            # concealed kong from hand
            agent.update(f"Player {_seat} AnGang {rp[1]}")
        elif rp[0] == "BUGANG":
            agent.update(f"Player {_seat} BuGang {rp[1]}")
        dbg(line, resp)
        respond(resp)

    elif rt == "3":
        pid = int(parts[1]); action = parts[2]
        tile1 = parts[3] if len(parts) > 3 else None

        # My own action echo — already applied at decision time. Never re-apply.
        if pid == _seat:
            respond("PASS"); return

        if action == "PLAY":
            _last_discard = tile1
            feat(line)                  # feat now offers Hu/Peng/Chi/Gang/Pass
            resp = decide_claim(tile1)
            resp = verify_claim(resp, tile1)         # universal legality guard
            if resp.startswith("PENG"):
                agent.update(f"Player {_seat} Peng")
                agent.update(f"Player {_seat} Play {resp.split()[1]}")
            elif resp.startswith("CHI"):
                p = resp.split()
                agent.update(f"Player {_seat} Chi {p[1]}")
                agent.update(f"Player {_seat} Play {p[2]}")
            elif resp == "GANG":
                agent.update(f"Player {_seat} Gang")  # exposed kong from discard
                _recent_kong = True
            dbg(line, resp)
            respond(resp)
        elif action == "BUGANG":
            _last_discard = tile1
            feat(line)
            resp = decide_robkong(tile1)
            dbg(line, resp)
            respond(resp)
        else:                            # DRAW / GANG / PENG / CHI by opponent
            feat(line)
            respond("PASS")
    else:
        respond("PASS")


# ── one-shot JSON (Botzone upload / run_match.py) ─────────────────────────────

def run_json_oneshot(data):
    import json
    turn_id = len(data.get("responses", []))
    reqs  = [data["requests"][i]  for i in range(turn_id + 1)]
    resps = [data["responses"][i] for i in range(turn_id)]
    p0 = reqs[0].split(); reset(int(p0[1]), int(p0[2]))

    # Replay history through feat_agent (authoritative state)
    for i in range(1, turn_id):
        req, resp = reqs[i], resps[i]
        parts = req.split(); rt = parts[0]
        if rt == "1":
            feat(req)
        elif rt == "2":
            feat(req)
            rp = resp.split()
            if rp[0] == "PLAY":
                agent.update(f"Player {_seat} Play {rp[1]}")
            elif rp[0] == "GANG":
                agent.update(f"Player {_seat} AnGang {rp[1]}")  # concealed kong
            elif rp[0] == "BUGANG":
                agent.update(f"Player {_seat} BuGang {rp[1]}")
        elif rt == "3":
            pid = int(parts[1]); action = parts[2]
            tile1 = parts[3] if len(parts) > 3 else None
            if pid == _seat:
                continue   # my own echo — already applied at decision time
            feat(req)
            rp = resp.split()
            if rp[0] == "PENG":
                agent.update(f"Player {_seat} Peng")
                agent.update(f"Player {_seat} Play {rp[1]}")
            elif rp[0] == "CHI":
                agent.update(f"Player {_seat} Chi {rp[1]}")
                agent.update(f"Player {_seat} Play {rp[2]}")
            elif rp[0] == "GANG" and action == "PLAY":
                agent.update(f"Player {_seat} Gang")

    # Current decision
    curr = reqs[turn_id]; parts = curr.split(); rt = parts[0]
    resp = "PASS"
    if rt == "2":
        feat(curr); resp = verify_draw(decide_draw(parts[1]), parts[1])
    elif rt == "3":
        pid = int(parts[1]); action = parts[2]
        tile1 = parts[3] if len(parts) > 3 else None
        if pid == _seat:
            resp = "PASS"
        elif action == "PLAY":
            feat(curr); resp = verify_claim(decide_claim(tile1), tile1)
        elif action == "BUGANG":
            feat(curr); resp = decide_robkong(tile1)
        else:
            feat(curr); resp = "PASS"
    print(__import__("json").dumps({"response": resp, "debug": env_status()}))


def run():
    first = sys.stdin.readline()
    if not first:
        return
    first = first.strip()
    if first.startswith("{"):
        import json
        run_json_oneshot(json.loads(first + sys.stdin.read()))
        return
    if first != "1" and first:
        handle(first)
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if line:
            try:
                handle(line)
            except Exception as e:
                if _dbg: _dbg.write(f"ERROR: {e}\n"); _dbg.flush()
                respond("PASS")


if __name__ == "__main__":
    run()
