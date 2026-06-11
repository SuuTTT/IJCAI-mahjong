# determinize.py — sample consistent hidden states for test-time PIMC search (opt-in CAIEST_PIMC).
#
# From the bot's OWN view (FeatureAgent: my hand=packs[0]/self.hand, everyone's melds=packs[4],
# everyone's discards=history[4], wall counts=tileWall[4]), reconstruct the UNSEEN multiset and
# deal it out to the 3 opponents' concealed hands (correct sizes) + the wall. Each sample is a
# full perfect-information world consistent with everything we've observed; PIMC averages over many.
#
# Pure tile-bookkeeping — no MahjongGB, no torch — so it unit-tests offline (tile conservation).
import random
from collections import defaultdict

TILE_LIST = [
    *('W%d' % (i + 1) for i in range(9)),
    *('T%d' % (i + 1) for i in range(9)),
    *('B%d' % (i + 1) for i in range(9)),
    *('F%d' % (i + 1) for i in range(4)),
    *('J%d' % (i + 1) for i in range(3)),
]


def _pack_tiles(tri):
    """Expand one meld (type, tile, offer) into the concrete tiles it consumes — matches feature.py."""
    typ, tile, _ = tri
    if typ == 'CHI':
        c, n = tile[0], int(tile[1:])
        return ['%s%d' % (c, n - 1), tile, '%s%d' % (c, n + 1)]
    if typ == 'PENG':
        return [tile] * 3
    if typ == 'GANG':
        return [tile] * 4
    return []


def concealed_size(packs_p):
    """Concealed (in-hand) tile count for a player with these melds: 13 - 3 per meld set.
    CHI/PENG/GANG each occupy one set slot; GANG's 4th tile is offset by its replacement draw."""
    return 13 - 3 * len(packs_p)


def unseen_pool(my_hand, packs, history):
    """The multiset of tiles NOT visible to us: full deck (4x34) minus my hand, minus ALL melds
    (every player), minus ALL discards (every player). Returns dict tile->count (>=0)."""
    pool = defaultdict(int)
    for t in TILE_LIST:
        pool[t] = 4
    for t in my_hand:
        pool[t] -= 1
    for p in range(4):
        for tri in packs[p]:
            for t in _pack_tiles(tri):
                pool[t] -= 1
        for t in history[p]:
            pool[t] -= 1
    return pool


def unseen_pool_shown(my_hand, shown):
    """Robust unseen pool from the bot's own visible-tile tally `shown` (FeatureAgent.shownTiles:
    discards + all melds, claim-bookkeeping correct). unseen[t] = 4 - shown[t] - my_hand.count(t)
    = opponents' concealed tiles + wall. Preferred over the packs/history reconstruction (which can
    double-count a discard that was later claimed into a meld)."""
    pool = defaultdict(int)
    myc = defaultdict(int)
    for t in my_hand:
        myc[t] += 1
    for t in TILE_LIST:
        pool[t] = max(0, 4 - int(shown.get(t, 0)) - myc[t])
    return pool


def sample_shown(my_hand, packs, shown, rng=random):
    """One determinization using the robust `shown`-based unseen pool. Opponent concealed sizes from
    packs. Returns {hands:{0..3}, wall:[...]} (hands[0] = our real hand)."""
    pool = unseen_pool_shown(my_hand, shown)
    bag = []
    for t, c in pool.items():
        if c > 0:
            bag.extend([t] * c)
    rng.shuffle(bag)
    hands = {0: list(my_hand)}
    i = 0
    for p in (1, 2, 3):
        k = concealed_size(packs[p])
        hands[p] = bag[i:i + k]
        i += k
    return dict(hands=hands, wall=bag[i:])


def check_consistency(my_hand, packs, history, tileWall):
    """Sanity: unseen total must equal sum(opp concealed sizes) + wall. Returns (ok, detail)."""
    pool = unseen_pool(my_hand, packs, history)
    neg = {t: c for t, c in pool.items() if c < 0}
    total_unseen = sum(max(c, 0) for c in pool.values())
    opp_concealed = sum(concealed_size(packs[p]) for p in (1, 2, 3))
    wall = sum(tileWall)
    ok = (not neg) and (total_unseen == opp_concealed + wall)
    return ok, dict(unseen=total_unseen, opp_concealed=opp_concealed, wall=wall, negative=neg)


def sample(my_hand, packs, history, tileWall, rng=random):
    """Draw ONE determinization. Returns dict:
        hands[p]  -> list of concealed tiles for each opponent p in {1,2,3} (p=0 is our real hand)
        wall      -> list of remaining wall tiles (order randomized; length=sum(tileWall))
    Distributes the unseen multiset uniformly at random into the correct-sized buckets."""
    pool = unseen_pool(my_hand, packs, history)
    bag = []
    for t, c in pool.items():
        if c > 0:
            bag.extend([t] * c)
    rng.shuffle(bag)
    hands = {0: list(my_hand)}
    i = 0
    for p in (1, 2, 3):
        k = concealed_size(packs[p])
        hands[p] = bag[i:i + k]
        i += k
    wall = bag[i:]
    return dict(hands=hands, wall=wall)


# ---- offline self-test: conservation under a synthetic mid-game state ----------------------------
if __name__ == '__main__':
    # me (seat0): 13 concealed, no melds. opp1 has a PENG (concealed 10). opp2 a CHI (10). opp3 none(13).
    my_hand = ['W1', 'W1', 'W2', 'W3', 'T5', 'T6', 'T7', 'B2', 'B3', 'B4', 'F1', 'F1', 'J3']
    packs = [
        [],
        [('PENG', 'W9', 1)],
        [('CHI', 'T5', 2)],
        [],
    ]
    history = [
        ['B9', 'J2'],
        ['F4', 'B1'],
        ['W6'],
        ['T1', 'T2', 'F3'],
    ]
    # wall counts must equal unseen - opp_concealed. compute then assert.
    pool = unseen_pool(my_hand, packs, history)
    total_unseen = sum(max(c, 0) for c in pool.values())
    opp = sum(concealed_size(packs[p]) for p in (1, 2, 3))
    wall_total = total_unseen - opp
    tileWall = [wall_total // 4 + (1 if i < wall_total % 4 else 0) for i in range(4)]
    ok, detail = check_consistency(my_hand, packs, history, tileWall)
    print('consistency:', ok, detail)
    s = sample(my_hand, packs, history, tileWall)
    sizes = {p: len(s['hands'][p]) for p in range(4)}
    print('hand sizes:', sizes, 'wall len:', len(s['wall']))
    # verify every tile type used <=4 across the full reconstructed world
    cnt = defaultdict(int)
    for p in range(4):
        for t in s['hands'][p]:
            cnt[t] += 1
    for t in s['wall']:
        cnt[t] += 1
    for p in range(4):
        for tri in packs[p]:
            for t in _pack_tiles(tri):
                cnt[t] += 1
    for p in range(4):
        for t in history[p]:
            cnt[t] += 1
    bad = {t: c for t, c in cnt.items() if c != 4}
    print('all tiles conserved (each type ==4):', not bad, '' if not bad else bad)
    assert ok and not bad, 'determinizer FAILED conservation'
    print('OK determinizer self-test passed')
