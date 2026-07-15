"""
featplus.py — mechanism-grounded EXTRA feature planes for the caiest (38,4,9) encoder.

All extra planes are derived FROM the existing 38-plane obs (so train/serve are guaranteed
consistent and re-cooking is just a transform of the already-cooked tensors). Plane layout of
the base 38 (feature.py):
    0      SEAT_WIND        (one-hot at this seat's wind tile)
    1      PREVALENT_WIND
    2..5   HAND             (count-encoded: k-th plane set if hand has >=k copies)
    6..21  DISCARD          4 seats x 4 planes  (seat 0=self, count-encoded discards)
    22..37 HALF_FLUSH/melds 4 seats x 4 planes  (revealed pung/chi/gang tiles)

Grid: (4,9) -> 36 positions, tile index = row*9+col. Rows: W(0) T(1) B(2) honors(3, F1-4 J1-3,
positions 27..33; 34,35 unused). This matches suit_aug (it permutes rows 0,1,2 = number suits),
so ALL extra planes below are computed per-suit-row and stay label-preserving under suit_aug.

Feature SETS (additive plane groups). Each returns (N, P, 4, 9) float32 in {0,1} or [0,1].

  A  danger/discard  (+5): per-opponent discard-count (3 planes, scalar broadcast of how many
                            tiles each opp has discarded -> "how deep into the game / how active")
                            + per-opponent meld-count (broadcast) ... actually we do it per-plane:
        A1  opp1 genbutsu? NO -> A is DISCARD-DERIVED danger:
        A planes: for each opponent o in {1,2,3}: a (4,9) plane = 1 where opp o has discarded
                  that tile AT ALL (binary "seen in o's river") -> 3 planes.
        + 2 planes: per-opponent meld COUNT broadcast (committed?) for opp-aggregate, and
                    total discards so far (game-progress) broadcast.

  B  fan/shanten   (+4): regular/7pairs/13orphans shanten of own concealed hand, each broadcast
                         as a (4,9) plane scaled 1-shanten/8 (clamped >=0); + a "useful tile" plane
                         (1 at every tile that reduces regular shanten if drawn). REQUIRES MahjongGB.

  C  genbutsu      (+3): for each opponent o in {1,2,3}, a (4,9) plane = 1 at every tile that is
                         genbutsu (present in o's discard river) -> provably safe vs rong (furiten).
                         This is the safe_discard signal made dense per-tile.  (Pure obs transform.)

Public API:
  add_planes_from_obs(obs, sets)         # obs (N,38,4,9) -> extra (N,P,4,9); fast/vectorised for A,C
  add_planes_one(obs1, sets)             # single (38,4,9) for the sim (runtime)
  PLANES = {'A':5,'B':4,'C':3}
"""
import numpy as np

PLANES = {'A': 5, 'B': 4, 'C': 3}

# tile-position helpers: 34 real tiles map to grid positions 0..33 (row*9+col), 34/35 unused.
_NUM = 27  # W,T,B number tiles occupy positions 0..26
# discard planes: base offset 6, 4 planes per seat; seat o occupies 6+4*o .. 6+4*o+3
# "discarded at all" by seat o  ==  any of its 4 discard planes set (plane k = >=k copies)
def _discard_any(obs, seat):
    # obs: (N,38,4,9) -> (N,4,9) binary: seat discarded this tile at least once
    base = 6 + 4 * seat
    return (obs[:, base:base + 4, :, :].sum(1) > 0).astype(np.float32)

def _meld_any(obs, seat):
    base = 22 + 4 * seat
    return (obs[:, base:base + 4, :, :].sum(1) > 0).astype(np.float32)

def _hand_counts_from_obs(o1):
    # o1: (38,4,9) -> length-34 int hand counts (count-encoded planes 2..5)
    h = o1[2:6].sum(0).reshape(-1)[:34]   # sum of the 4 count planes = #copies
    return h.astype(np.int64)

def planes_A(obs):
    N = obs.shape[0]
    out = np.zeros((N, 5, 4, 9), np.float32)
    # 3 planes: each opponent's "seen in river" (binary)
    for i, o in enumerate((1, 2, 3)):
        out[:, i] = _discard_any(obs, o)
    # plane 3: opponent commitment = sum over opp melds (any) -> broadcast scalar per sample
    melds = sum(_meld_any(obs, o).reshape(N, -1).max(1) for o in (1, 2, 3))  # 0..3 opp committed-ish
    out[:, 3] = (melds / 3.0)[:, None, None]
    # plane 4: game progress = total discards seen / 60 (broadcast scalar)
    tot = sum(_discard_any(obs, o).reshape(N, -1).sum(1) for o in (0, 1, 2, 3))
    out[:, 4] = np.clip(tot / 60.0, 0, 1)[:, None, None]
    return out

def planes_C(obs):
    N = obs.shape[0]
    out = np.zeros((N, 3, 4, 9), np.float32)
    for i, o in enumerate((1, 2, 3)):
        out[:, i] = _discard_any(obs, o)   # genbutsu == in that opp's river (furiten => can't rong it)
    return out

# ---- B: shanten / useful-tile (needs MahjongGB; per-sample) ----
_TILES34 = ([f"W{n}" for n in range(1, 10)] + [f"T{n}" for n in range(1, 10)] +
            [f"B{n}" for n in range(1, 10)] + [f"F{n}" for n in range(1, 5)] + [f"J{n}" for n in range(1, 4)])
def _pos_of_tile(idx):
    # tile idx 0..33 -> grid (row,col); rows W/T/B/honors
    return idx  # we fill via reshape on length-36

def _hand_tiles_from_counts(cnt):
    out = []
    for i, c in enumerate(cnt):
        out += [_TILES34[i]] * int(c)
    return out

def _shanten_planes_one(o1, calc):
    # returns (4,4,9): reg, 7p, 13o shanten broadcast + useful-tile plane
    from MahjongGB import RegularShanten, SevenPairsShanten, ThirteenOrphansShanten
    cnt = _hand_counts_from_obs(o1)
    tiles = _hand_tiles_from_counts(cnt)
    out = np.zeros((4, 4, 9), np.float32)
    # shanten only well-defined for concealed-ish hand sizes (13/10/7/4/1 + draw). MahjongGB needs
    # 3k+1 or 3k+2 concealed tiles; if the count is off (post-meld), we degrade to a big shanten.
    n = len(tiles)
    reg = sp = to = 8
    useful = []
    if n in (1, 2, 4, 5, 7, 8, 10, 11, 13, 14):
        try:
            reg, u = RegularShanten(tuple(tiles)); useful = list(u)
        except Exception:
            reg = 8
        try:
            sp, _ = SevenPairsShanten(tuple(tiles))
        except Exception:
            sp = 8
        try:
            to, _ = ThirteenOrphansShanten(tuple(tiles))
        except Exception:
            to = 8
    out[0] = max(0.0, (8 - min(reg, 8)) / 8.0)
    out[1] = max(0.0, (8 - min(sp, 8)) / 8.0)
    out[2] = max(0.0, (8 - min(to, 8)) / 8.0)
    flat = np.zeros(36, np.float32)
    for t in useful:
        if t in _TILES34:
            flat[_TILES34.index(t)] = 1.0
    out[3] = flat.reshape(4, 9)
    return out

def planes_B(obs, nproc=1):
    N = obs.shape[0]
    out = np.zeros((N, 4, 4, 9), np.float32)
    for i in range(N):
        out[i] = _shanten_planes_one(obs[i], None)
    return out

def add_planes_from_obs(obs, sets):
    parts = []
    if 'A' in sets: parts.append(planes_A(obs))
    if 'B' in sets: parts.append(planes_B(obs))
    if 'C' in sets: parts.append(planes_C(obs))
    return np.concatenate(parts, axis=1) if parts else np.zeros((obs.shape[0], 0, 4, 9), np.float32)

def add_planes_one(o1, sets):
    return add_planes_from_obs(o1[None], sets)[0]

def total_extra(sets):
    return sum(PLANES[s] for s in sets)
