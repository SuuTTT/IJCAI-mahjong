"""
winprob_controller.py — WITHIN-GAME WIN-RATE controller (drop-in for choose_discard).

Lever (DISTINCT from targeted_defense.py's danger trigger): play the RAW aggressive policy by
default; override the discard to a SAFE (genbutsu) tile ONLY when P(I win THIS hand) < TAU — i.e. the
hand is DEAD/hopeless. Hypothesis: folding a hand that won't win anyway sheds ~no expected offense,
while still dodging the expensive (-40) rong deal-ins => could be net-positive on placement, unlike
the danger trigger which folds LIVE (winning) hands too.

Genbutsu = furiten-safe (a tile already in opp p's discards can never ron p). We swap to the highest-
policy-ranked tile that is safe vs all strongly-committed opponents, within the top-K (preserving most
offense). No safe tile in top-K, or no committed opponent -> keep raw (no-op).

Drop-in: same signature as safe_discard.choose_discard / targeted_defense.choose_discard. Globals
TAU/TOP_K/MIN_TURN/MIN_MELDS set by the gate per-worker.
"""
import os
import winprob_estimator as WP

TAU      = float(os.environ.get("WP_TAU", "0.15"))   # fold when P(win) < TAU (dead hand)
TOP_K    = int(os.environ.get("WP_TOPK", "4"))       # search depth for a safe discard
MIN_TURN = int(os.environ.get("WP_MIN_TURN", "6"))   # don't fold in the opening
MIN_MELDS = int(os.environ.get("WP_MIN_MELDS", "1")) # only defend vs committed opponents (>=1 meld)

STATS = {"discards": 0, "fired": 0, "noop_same": 0, "wp_sum": 0.0, "wp_fired_sum": 0.0,
         "low_wp": 0}  # low_wp = #discards where P(win)<TAU (the fold TRIGGER fired, pre-safe-check)


def choose_discard(agent, ranked_tiles, top_k=None):
    k = TOP_K if top_k is None else top_k
    STATS["discards"] += 1
    raw = ranked_tiles[0]
    if len(agent.history[0]) < MIN_TURN:
        STATS["noop_same"] += 1
        return raw
    p_win = WP.win_prob(agent)
    STATS["wp_sum"] += p_win
    if p_win >= TAU:
        STATS["noop_same"] += 1
        return raw                       # hand still alive -> stay aggressive (== raw)
    STATS["low_wp"] += 1
    # dead hand: try to swap to a genbutsu (safe) tile within the top-K, else keep raw
    threats = [pl for pl in (1, 2, 3) if len(agent.packs[pl]) >= MIN_MELDS]
    if not threats:
        STATS["noop_same"] += 1
        return raw                       # no committed opp to defend against -> no-op
    def genbutsu(t):
        return all(t in agent.history[pl] for pl in threats)
    cands = ranked_tiles[:k] if k > 0 else ranked_tiles
    for t in cands:
        if genbutsu(t):
            if t == raw:
                STATS["noop_same"] += 1
                return raw               # raw is already safe -> no-op
            STATS["fired"] += 1
            STATS["wp_fired_sum"] += p_win
            return t
    STATS["noop_same"] += 1
    return raw                           # nothing safe in top-K -> keep offense (no-op)
