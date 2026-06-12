# pimc_obs.py — standalone per-seat observation encoder for net-driven PIMC rollouts.
#
# Mirrors feature.py's obs layout EXACTLY (validated by test_obs_parity.py replaying real games):
#   plane 0      SEAT_WIND       one-hot F{seatWind+1}
#   plane 1      PREVALENT_WIND  one-hot F{prevalentWind+1}
#   planes 2-5   HAND            thermometer of concealed-hand counts
#   planes 6-21  DISCARD         4 planes per RELATIVE position (0=self,1=next,...) thermometer
#   planes 22-37 HALF_FLUSH      4 planes per relative position: meld-expanded tile counts
#                                (CHI -> [t-1,t,t+1], PENG -> [t]x3, GANG -> [t]x4)
# Output (38, 4, 9) float32 — what the fused policy/value nets consume.
#
# Rollout state is OUR-relative (index 0 = us). For acting seat r, r's relative position j maps to
# our index (r + j) % 4, and r's seatWind = seatwinds[r].
import numpy as np

TILE_LIST = [
    *('W%d' % (i + 1) for i in range(9)),
    *('T%d' % (i + 1) for i in range(9)),
    *('B%d' % (i + 1) for i in range(9)),
    *('F%d' % (i + 1) for i in range(4)),
    *('J%d' % (i + 1) for i in range(3)),
]
TILE_IDX = {t: i for i, t in enumerate(TILE_LIST)}
OBS_SIZE, HAND, DISCARD, FLUSH = 38, 2, 6, 22


def _thermo(obs, base, tiles):
    """Set thermometer planes obs[base:base+count, col] = 1 for each tile's multiset count."""
    cnt = {}
    for t in tiles:
        cnt[t] = cnt.get(t, 0) + 1
    for t, c in cnt.items():
        obs[base:base + min(c, 4), TILE_IDX[t]] = 1


def _meld_tiles(packs_p):
    out = []
    for typ, tile, _ in packs_p:
        if typ == 'CHI':
            c, n = tile[0], int(tile[1:])
            out += ['%s%d' % (c, n - 1), tile, '%s%d' % (c, n + 1)]
        elif typ == 'PENG':
            out += [tile] * 3
        elif typ == 'GANG':
            out += [tile] * 4
    return out


def obs_for_seat(r, hands, discards, packs, seatwinds, prevalent):
    """Seat r's observation of an our-relative rollout state.
    hands: dict/list of concealed-tile lists; discards: list of 4 discard lists; packs: 4 meld lists;
    seatwinds[r]: r's wind; all indexed our-relative. Returns (38,4,9) float32."""
    obs = np.zeros((OBS_SIZE, 36), np.float32)
    obs[0, TILE_IDX['F%d' % (seatwinds[r] + 1)]] = 1
    obs[1, TILE_IDX['F%d' % (prevalent + 1)]] = 1
    _thermo(obs, HAND, hands[r])
    for j in range(4):                       # j = relative-to-r position (0=r itself)
        src = (r + j) % 4
        _thermo(obs, DISCARD + 4 * j, discards[src])
        _thermo(obs, FLUSH + 4 * j, _meld_tiles(packs[src]))
    return obs.reshape(OBS_SIZE, 4, 9)


def legal_play_mask(hand):
    """235-mask with only Play actions for tiles in hand (rollouts are discard-only)."""
    m = np.zeros(235, np.float32)
    for t in set(hand):
        m[2 + TILE_IDX[t]] = 1
    return m
