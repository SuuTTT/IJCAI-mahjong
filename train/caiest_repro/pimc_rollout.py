# pimc_rollout.py — cheap-playout determinized rollout for test-time PIMC search.
#
# Reuses csm_rollout (verified _fan_count / MCR scoring / conversion-biased playout) + pimc_obs
# (byte-exact 38-plane leaf encoder) + determinize (consistent hidden-state sampling).
#
# One rollout from a determinized world: after OUR candidate discard, play ALL FOUR seats forward
# with the cheap fan/greedy playout, checking self-draw Hu (on draw) and robbing Hu (on discard),
# for up to H plies. If a Hu ends the game -> return ('term', our avg_rank in [1,4]) = the EXACT
# duplicate placement of our seat (same rank formula the gate uses). If truncated at H -> return
# ('leaf', our (38,4,9) obs) so the caller can BATCH the V_place head (predicted final placement,
# also in [1,4]). Both branches are in placement units, LOWER = better -> a single consistent
# objective the search minimizes. Discards are seeded with the REAL per-seat discard history so the
# leaf obs's DISCARD planes are correct (not just rollout-generated).
import os
from collections import Counter
import numpy as np

import csm_rollout as CR
import pimc_obs as PO

ROLL = os.environ.get('CAIEST_PIMC_ROLL', 'fan')   # 'fan' = conversion-biased playout, else greedy


def _avg_rank(scores, me=0):
    """Duplicate placement of seat `me` from a 4-score vector (matches e8_gate's rank formula)."""
    greater = sum(1 for j in range(4) if scores[j] > scores[me])
    equal = sum(1 for j in range(4) if scores[j] == scores[me])
    return greater + (equal + 1) / 2.0


def rollout_leaf(my_hand_after, discard_tile, world, packs, seatwinds, prevalent, base_discards, shown0, H, rng):
    """my_hand_after: our concealed hand AFTER the candidate discard (index 0). discard_tile: the
    tile we just discarded (checked for immediate opponent rong = the deal-in signal). world:
    determinized {hands, wall}. packs/seatwinds our-relative. base_discards: real per-seat discard
    lists (seed). shown0: aggregate tile->count seen (for is4th). H: horizon plies. Returns
    ('term', avg_rank) or ('leaf', obs38x4x9)."""
    hands = [list(my_hand_after), list(world['hands'][1]), list(world['hands'][2]), list(world['hands'][3])]
    discards = [list(base_discards[j]) for j in range(4)]
    wall = list(world['wall'])
    shown = Counter(shown0)
    # --- immediate robbing Hu on OUR just-made discard (the safe-discard / deal-in signal) ---
    is4d0 = shown.get(discard_tile, 0) == 4
    wallLast0 = len(wall) <= 1
    for r in (1, 2, 3):
        fc = CR._fan_count(packs[r], hands[r], discard_tile, False, seatwinds[r], prevalent, is4d0, wallLast0)
        if fc:
            return ('term', _avg_rank(CR._scores_discard_win(r, 0, fc)))   # we (seat 0) deal in
    discards[0].append(discard_tile); shown[discard_tile] = shown.get(discard_tile, 0) + 1
    cur = 1                                            # after our discard, player 1 acts
    for _ply in range(H):
        if not wall:
            return ('term', _avg_rank([0, 0, 0, 0]))   # exhaustive draw -> all tie -> 2.5
        wallLast = len(wall) <= 1
        t = wall.pop(); hands[cur].append(t)
        is4 = shown.get(t, 0) == 4
        he = list(hands[cur]); he.remove(t)
        fc = CR._fan_count(packs[cur], he, t, True, seatwinds[cur], prevalent, is4, wallLast)
        if fc:
            return ('term', _avg_rank(CR._scores_self_draw(cur, fc)))
        if ROLL == 'fan':
            d = CR._convert_discard(hands[cur], packs[cur], rng)
        else:
            d = CR._greedy_discard(hands[cur], packs[cur], rng)
        hands[cur].remove(d); discards[cur].append(d); shown[d] = shown.get(d, 0) + 1
        is4d = shown[d] == 4
        for r in range(4):
            if r == cur:
                continue
            fc = CR._fan_count(packs[r], hands[r], d, False, seatwinds[r], prevalent, is4d, wallLast)
            if fc:
                return ('term', _avg_rank(CR._scores_discard_win(r, cur, fc)))
        cur = (cur + 1) % 4
    obs = PO.obs_for_seat(0, hands, discards, packs, seatwinds, prevalent)
    return ('leaf', obs)
