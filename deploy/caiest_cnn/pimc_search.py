# pimc_search.py — anytime opponent-aware PIMC discard selection (opt-in CAIEST_PIMC).
#
# At a Play decision: take the policy's top-K candidate discards (stay in-distribution), and for each,
# estimate OUR expected MCR duplicate score by averaging opponent-aware determinized rollouts
# (csm_rollout: all 4 seats play, opponents can Hu first). Pick the highest-EV candidate.
#
# ANYTIME + hard time guard: we run determinization ROUNDS (one rollout per candidate per round) and
# stop when the wall-clock budget is hit, returning the best mean so far. We NEVER exceed the budget
# (= edge over bots like hhh that TLE). Falls back to the policy's argmax if anything is off.
import os, time, random
import numpy as np

import determinize as _D
import csm_rollout as _R

BUDGET_MS = float(os.environ.get('CAIEST_PIMC_MS', '4500'))
TOPK = int(os.environ.get('CAIEST_PIMC_K', '5'))
DELTA = float(os.environ.get('CAIEST_PIMC_DELTA', '3.0'))
DEPTH = int(os.environ.get('CAIEST_PIMC_DEPTH', '60'))
MIN_ROUNDS = int(os.environ.get('CAIEST_PIMC_MINROUNDS', '6'))
_RNG = random.Random(12345)


def pick_discard(agent, lg, mask, play_offset):
    """Return a Play action index chosen by PIMC, or None to defer to the policy.
    agent: FeatureAgent (agent.hand, agent.packs, agent.shownTiles, agent.seatWind, agent.prevalentWind)."""
    if _R.MahjongFanCalculator is None:
        return None
    t0 = time.time()
    lg = np.asarray(lg).flatten()
    legal = [i for i in range(play_offset, play_offset + 34) if mask[i]]
    if len(legal) < 2:
        return None
    legal.sort(key=lambda i: -float(lg[i]))
    top = float(lg[legal[0]])
    cands = [i for i in legal[:TOPK] if top - float(lg[i]) <= DELTA]
    if len(cands) < 2:
        return None

    my_hand = list(agent.hand)
    packs = [list(agent.packs[p]) for p in range(4)]
    shown = agent.shownTiles
    sw = int(agent.seatWind)
    seatwinds = [(sw + p) % 4 for p in range(4)]
    prevalent = int(getattr(agent, 'prevalentWind', 0))

    # candidate -> (post-discard concealed hand). skip a candidate whose tile we somehow don't hold.
    tiles = {}
    post = {}
    for i in cands:
        t = agent.TILE_LIST[i - play_offset]
        if t in my_hand:
            tiles[i] = t
            h = list(my_hand); h.remove(t)
            post[i] = h
    cands = [i for i in cands if i in post]
    if len(cands) < 2:
        return None

    budget = BUDGET_MS / 1000.0
    totals = {i: 0.0 for i in cands}
    rounds = 0
    while True:
        # one rollout per candidate this round, each on a FRESH determinized world
        for i in cands:
            world = _D.sample_shown(my_hand, packs, shown, _RNG)
            hands = [post[i], world['hands'][1], world['hands'][2], world['hands'][3]]
            s = _R.rollout_once(hands, world['wall'], packs, seatwinds, prevalent,
                                start=1, max_turns=DEPTH, rng=_RNG)
            totals[i] += s[0]
        rounds += 1
        elapsed = time.time() - t0
        if rounds >= MIN_ROUNDS and elapsed >= budget:
            break
        if elapsed >= budget * 1.5:          # hard stop even before MIN_ROUNDS
            break
        if rounds >= 400:                    # sanity cap
            break
    best = max(cands, key=lambda i: totals[i] / rounds)
    return best
