"""
winprob_estimator.py — heuristic P(I win THIS hand) from the live FeatureAgent state, for the
IJCAI-2026 Mahjong (MCR) WITHIN-GAME win-rate controller.

DISTINCT from targeted_defense.py (which estimates P(deal-in)*E[opp fan] — an OPPONENT-side danger
signal). This estimates MY OWN current-hand offense: how close am I to completing a >=8-fan win,
given my concealed hand + melds, the live copies of the tiles I still need, and how many draws I have
left. The controller folds (genbutsu) ONLY when this is LOW (a dead/hopeless hand) — the hypothesis
being that folding a hand that won't win anyway sheds ~no offense while still dodging the -40 deal-ins.

No trained value/critic checkpoint exists on the box (searched ckpt/ — only policy nets and the
score-regression value_head.py scaffold, which has NO labeled data file). The cooked 5.87M dataset
carries only (obs,mask,act) — NO win/score labels — so a clean trained P(win) head would require a
full re-cook from match logs OR self-play harvesting. We therefore use a HEURISTIC computed from the
SAME FeatureAgent state available at deploy time (zero train/serve skew), and we VALIDATE its
calibration (AUC / Brier) against GROUND-TRUTH win labels harvested from raw self-play (see
winprob_calibrate.py). The honest claim is only about whether WIN-PROB-GATED folding beats raw on
placement — not that these probabilities are exactly correct.

State (FeatureAgent, p=0 == self):
  agent.hand          : my concealed tiles (list of "W3","F1",...). At a discard turn this is the
                        post-draw hand (3k+2 tiles); RegularShanten needs a 3k+1 standing hand, so
                        we take the MIN shanten over discarding each tile (= shanten of the 3k+2 hand).
  agent.packs[0]      : my revealed melds [(type,tile,offer)] — implied by the standing-tile count.
  agent.shownTiles[t] : copies of t already visible anywhere (4 - this = live copies unseen).
  agent.tileWall[0]   : tiles left in MY draw wall == draws I have left this hand.
"""
import os
from functools import lru_cache

try:
    from MahjongGB import RegularShanten, SevenPairsShanten, ThirteenOrphansShanten
    _HAS = True
except Exception:
    _HAS = False


def _shanten_standing(hand_tuple):
    """Min shanten over Regular / SevenPairs / ThirteenOrphans for a 3k+1 standing hand."""
    best = 99
    best_useful = ()
    for fn in (RegularShanten, SevenPairsShanten, ThirteenOrphansShanten):
        try:
            s, useful = fn(hand_tuple)
            if s < best:
                best = s
                best_useful = tuple(useful)
        except Exception:
            pass
    return best, best_useful


@lru_cache(maxsize=200000)
def _best_shanten_and_useful(hand_sorted):
    """For a post-draw hand (3k+2 tiles), return (min shanten after best discard, useful tiles).
    Tries dropping each distinct tile -> a 3k+1 standing hand -> shanten. Caches on sorted hand."""
    hand = list(hand_sorted)
    n = len(hand)
    if n % 3 == 1:
        # already a standing hand (shouldn't usually happen at our discard turn, but be safe)
        s, u = _shanten_standing(tuple(hand))
        return s, tuple(u)
    if n % 3 != 2:
        return 99, ()
    best = 99
    best_useful = ()
    for t in set(hand):
        rest = list(hand)
        rest.remove(t)
        s, u = _shanten_standing(tuple(rest))
        if s < best:
            best = s
            best_useful = tuple(u)
    return best, best_useful


def _live_copies(agent, tile):
    return max(0, 4 - agent.shownTiles.get(tile, 0))


def win_prob(agent):
    """Heuristic P(I complete a winning hand THIS hand) in [0,1].

    Model: I need (shanten+1) successful advancing draws (each lands one of my `useful` tiles) within
    `draws_left` draws. Per-draw advance prob p_adv ~ (live copies of useful tiles) / (unseen tiles).
    P(win) ~ P(>= shanten+1 advances in draws_left Bernoulli(p_adv) trials), discounted by a fan-floor
    factor (MCR needs >=8 fan; a bare hand that reaches "tenpai" may still be unscorable). Calibrated
    /validated against ground-truth win labels (winprob_calibrate.py).

    Already-won / 0-shanten (tenpai) hands get high prob; hopeless (high shanten, few draws, dead
    waits) get ~0 — exactly the dead-hand signal the controller folds on.
    """
    if not _HAS:
        return 0.5  # estimator unavailable -> neutral (controller will then ~never fold; honest)
    hand = agent.hand
    if not hand:
        return 0.0
    shanten, useful = _best_shanten_and_useful(tuple(sorted(hand)))
    if shanten >= 90:
        return 0.0
    draws_left = max(0, int(agent.tileWall[0]))
    # tiles that advance the hand, weighted by how many are still live (unseen)
    live_useful = sum(_live_copies(agent, t) for t in useful) if useful else 0
    # rough count of unseen tiles: 136 - (my 13/14) - everything shown. Use shownTiles + my hand.
    seen = sum(agent.shownTiles.values()) + len(hand)
    unseen = max(1, 136 - seen)
    p_adv = min(0.9, live_useful / unseen) if unseen > 0 else 0.0
    need = shanten + 1  # advances needed to reach a complete (0-shanten -> need 1 winning draw)
    if shanten == 0:
        need = 1
    # Also account for RON: opponents discard ~ (3 * draws_left) tiles I could ron. Treat ron as
    # extra "effective draws" with the same per-tile live-useful odds (a useful tile from anyone).
    eff_draws = draws_left + 3 * draws_left  # my draws + ~3 opp discards per round
    if eff_draws <= 0 or p_adv <= 0:
        base = 0.0
    else:
        # P(>= need successes in eff_draws Bernoulli(p_adv)) — compute via complement for small need.
        # For need<=4 this is cheap and exact enough.
        from math import comb
        q = 1.0 - p_adv
        p_lt = 0.0
        for k in range(0, min(need, eff_draws + 1)):
            p_lt += comb(eff_draws, k) * (p_adv ** k) * (q ** (eff_draws - k))
        base = max(0.0, 1.0 - p_lt)
    # MCR 8-fan floor: a hand with no melds toward yaku and tiny shanten advantage is still hard to
    # SCORE. Apply a mild floor-discount that fades as the hand gets richer (more melds / lower shanten).
    nmeld = len(agent.packs[0])
    fan_floor = 0.55 + 0.10 * nmeld + (0.15 if shanten == 0 else 0.0)
    fan_floor = min(1.0, fan_floor)
    return max(0.0, min(1.0, base * fan_floor))
