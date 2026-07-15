"""
targeted_defense.py — FAN-WEIGHTED, COST-GATED defensive overlay for IJCAI-2026 Mahjong (MCR).

Rationale (the precise lever the scoring asymmetry justifies):
  MCR deal-in is asymmetric. If YOU feed the rong winner you pay (fan + 8) for THAT hand — up to
  -40+ on a big hand — whereas a non-feeder pays a flat ~ -8. So the controllable extra loss is
  RONG-FEEDING, and the loss scales with the WINNER'S FAN. Blanket folding (frontier null) folded on
  P(deal-in) regardless of hand SIZE: it shed cheap deal-ins, lost a lot of 1st-rate, net <= raw.

  NEW trigger: play the raw aggressive policy by DEFAULT; override the discard to a SAFE (genbutsu)
  tile ONLY when the expected DEAL-IN COST of the raw choice exceeds tau:
        cost(t) = sum_p  P(deal-in to opp p | discard t)  x  (8 + E[fan_p])
  i.e. defend only to dodge the expensive (big-fan) deal-ins, stay aggressive everywhere else.

No learned DealinNet/WaitNet exists on the box, so P(deal-in) and E[fan] are HEURISTIC estimates
from observable state (opponent melds, discards, turn, live copies of t) + MahjongGB shanten. This is
the task's explicit fallback ("approximate E[opponent fan] from their melds/discards + tenpai-prob x
typical-fan"). The estimate is calibrated to be conservative and is documented as a heuristic, not a
trained predictor — the honest claim is only about whether COST-GATING the EXISTING genbutsu swap
(which the frontier already implements) beats raw, not that the cost numbers are exact.

Integration: drop-in for safe_discard.choose_discard signature. The gate calls choose_discard(agent,
ranked_tiles, top_k); we ignore mode and use module globals TAU, TOP_K, MIN_TURN.

agent = FeatureAgent: agent.history[p] = player p's discards (p relative, 0=self),
agent.packs[p] = melds [(type,tile,offer)], agent.shownTiles = visible counts, agent.tileWall[p].
"""
import os

TAU      = float(os.environ.get("TD_TAU", "3.0"))   # cost threshold (fan-points of expected rong feed)
TOP_K    = int(os.environ.get("TD_TOPK", "4"))      # search depth for a cheaper safe discard
MIN_TURN = int(os.environ.get("TD_MIN_TURN", "6"))  # don't bother early (no one is tenpai yet)

# instrumentation (per-process counters; the gate aggregates)
STATS = {"discards": 0, "fired": 0, "noop_same": 0, "cost_sum": 0.0, "cost_fired_sum": 0.0}


def _suit_flush_bonus(packs_p):
    """If all melded tiles are one suit -> flush-ish hand (Qing/Hun): big fan. Crude signal."""
    suits = set()
    honor = False
    for (_, tile, _) in packs_p:
        c = tile[0]
        if c in "WTB":
            suits.add(c)
        else:
            honor = True
    if not packs_p:
        return 0.0
    if len(suits) == 1:           # all number-tile melds same suit
        return 8.0 if not honor else 4.0   # toward Qing Yi Se(24) / Hun Yi Se(6)
    return 0.0


def _fan_estimate(agent, p):
    """E[fan] of opponent p's hand IF they win — heuristic from revealed melds/discards.
    Base hand ~ a few fan; pung of honors/terminals, declared gang, and one-suit melds add fan."""
    packs = agent.packs[p]
    fan = 1.0                         # a winning hand is >= 1 fan (chicken / pinghu floor-ish)
    for (typ, tile, _) in packs:
        c, n = tile[0], (int(tile[1]) if tile[1].isdigit() else 0)
        if typ in ("PENG", "GANG"):
            if c == "F" or c == "J":          # wind / dragon pung -> yaku, often doubles
                fan += 2.0
            elif c in "WTB" and (n == 1 or n == 9):  # terminal pung
                fan += 1.0
            else:
                fan += 0.5
        if typ == "GANG":
            fan += 1.0                          # gang itself is +fan and signals a big hand
    fan += _suit_flush_bonus(packs)
    # more melds = closer to a completed, scored hand -> nudge up
    fan += 0.5 * len(packs)
    return fan


def _tenpai_prob(agent, p):
    """P(opp p is tenpai / dangerously close) — heuristic from melds & turn.
    More melds + later game => higher. Bounded [0,1]. No melds & early => low."""
    nmeld = len(agent.packs[p])
    turn = len(agent.history[p])
    # melds drive commitment; each meld ~ +0.18, each discard past turn 6 ~ +0.03
    pr = 0.10 + 0.18 * nmeld + 0.03 * max(0, turn - 6)
    return max(0.0, min(0.95, pr))


def _live_copies(agent, tile):
    """How many copies of `tile` are NOT yet visible (in someone's hand or wall). 4 - shown."""
    shown = agent.shownTiles.get(tile, 0)
    return max(0, 4 - shown)


def _pdealin_one(agent, p, tile):
    """P(discarding `tile` deals into opp p). 0 if genbutsu (in p's discards => furiten-safe).
    Else tenpai_prob(p) x P(tile is one of p's winning waits). Wait-prob proxied by live copies
    (a tile with more live copies is a more plausible live wait) and a per-tile base rate."""
    if tile in agent.history[p]:
        return 0.0                       # genbutsu: 100% safe vs p (hard rule)
    tp = _tenpai_prob(agent, p)
    if tp <= 0.0:
        return 0.0
    live = _live_copies(agent, tile)
    if live <= 0:
        return 0.0                       # all 4 visible -> cannot be a live ron wait of p
    # base chance a given tile completes a specific tenpai hand's wait. A tenpai hand waits on
    # ~1-3 tile kinds out of ~34 -> ~0.05-0.10 prior; scale mildly by live copies (more live = more
    # plausibly still waited on, fewer = partly dead). Cap so it stays a probability.
    base = 0.07
    return min(0.6, tp * base * (1.0 + 0.25 * (live - 1)))


def deal_in_cost(agent, tile):
    """Expected RONG-FEED cost (in fan-points incl. the +8 base) of discarding `tile` right now."""
    cost = 0.0
    for p in (1, 2, 3):
        pdi = _pdealin_one(agent, p, tile)
        if pdi <= 0.0:
            continue
        cost += pdi * (8.0 + _fan_estimate(agent, p))
    return cost


def choose_discard(agent, ranked_tiles, top_k=None):
    """Targeted cost-gated defense. Play raw (ranked_tiles[0]) UNLESS its expected deal-in cost
    exceeds TAU; then swap to the LOWEST-COST tile within the top-K (preserving most offense)."""
    k = TOP_K if top_k is None else top_k
    STATS["discards"] += 1
    raw = ranked_tiles[0]
    if len(agent.history[0]) < MIN_TURN:
        STATS["noop_same"] += 1
        return raw
    raw_cost = deal_in_cost(agent, raw)
    STATS["cost_sum"] += raw_cost
    if raw_cost <= TAU:
        STATS["noop_same"] += 1
        return raw                       # cheap/safe enough -> stay aggressive (== raw)
    # high-cost state: find the cheapest discard among the policy's top-K (limit offense loss)
    cands = ranked_tiles[:k] if k > 0 else ranked_tiles
    best = min(cands, key=lambda t: deal_in_cost(agent, t))
    if best == raw:
        STATS["noop_same"] += 1
        return raw                       # nothing cheaper in top-K -> keep offense (no-op)
    STATS["fired"] += 1
    STATS["cost_fired_sum"] += raw_cost
    return best
