"""
dragon_aug.py — DRAGON-permutation symmetry (relabel the 3 dragons 中/發/白 = J1/J2/J3,
tile idx 31/32/33, obs row 3 cols 4/5/6) for Chinese Standard Mahjong. In MCR scoring the
three dragons are interchangeable (dragon pung = 1 fan each; little/big three dragons treat
them symmetrically), so relabeling dragons among themselves is a label-preserving 6x (3!)
augmentation. Winds (F1-4) are NOT permutable (seat/prevalent wind break their symmetry) and
are left fixed. Composes commutatively with suit-perm (number suits) and rank-reflection
(within-suit rank reversal) because it acts on a disjoint tile group (dragons only).

Action layout: Pass0 Hu1 Play[2+t] Chi[36+color*21+w] Peng[99+t] Gang[133+t] AnGang[167+t] BuGang[201+t].
Dragons have no Chi; only the tile-indexed blocks (Play/Peng/Gang/AnGang/BuGang) at t in {31,32,33}
are permuted. Row 3 obs cols: F1-4 -> cols 0-3, J1-3 -> cols 4-6 (cols 7,8 unused).
"""
import numpy as np
from itertools import permutations

OFF = {'Play': 2, 'Chi': 36, 'Peng': 99, 'Gang': 133, 'AnGang': 167, 'BuGang': 201}
DRAGON_IDX = (31, 32, 33)          # J1,J2,J3 tile indices
PERMS_D = [p for p in permutations(range(3))]   # 6 dragon permutations (incl identity)


def _tile_perm(q):
    """q: tuple mapping new dragon slot -> src dragon slot. Return length-34 tile remap
    dt[new_tile]=old_tile (only dragons 31,32,33 move; everything else fixed)."""
    dt = np.arange(34)
    for new in range(3):
        dt[31 + new] = 31 + q[new]
    return dt


def action_perm(q):
    """Return length-235 array A: A[new_action]=old_action (gather map for masks)."""
    dt = _tile_perm(q)
    A = np.arange(235)
    A[0] = 0; A[1] = 1
    for base in (OFF['Play'], OFF['Peng'], OFF['Gang'], OFF['AnGang'], OFF['BuGang']):
        for nt in range(34):
            A[base + nt] = base + dt[nt]
    # Chi block (36..98) left identity: dragons cannot be chi'd
    return A


def fwd_action_perm(q):
    """Return F: F[old_action]=new_action."""
    A = action_perm(q); F = np.empty(235, np.int64); F[A] = np.arange(235); return F


def obs_col_map(q):
    """length-9 gather for obs row-3 columns: new_row3[:,c]=old_row3[:,colmap[c]].
    Dragons at cols 4,5,6 (=slots 0,1,2) permuted; winds cols 0-3 and unused 7,8 fixed."""
    colmap = np.arange(9)
    for new in range(3):
        colmap[4 + new] = 4 + q[new]
    return colmap


def dragon_obs(o, q):
    """o (...,38,4,9) -> permute dragon columns in honor row (row idx 3); number rows unchanged."""
    out = o.copy()
    cm = obs_col_map(q)
    out[..., 3, :] = o[..., 3, cm]
    return out


if __name__ == '__main__':
    # self-tests: group structure + block permutation + involution of a transposition
    ids = PERMS_D
    for q in ids:
        A = action_perm(q); F = fwd_action_perm(q)
        assert np.array_equal(F[A], np.arange(235)), f"A/F not inverse for {q}"
        # each tile-block maps onto itself (no leakage across action types)
        for lo, hi in [(0, 1), (1, 2), (OFF['Play'], OFF['Play'] + 34), (OFF['Chi'], OFF['Chi'] + 63),
                       (OFF['Peng'], OFF['Peng'] + 34), (OFF['Gang'], OFF['Gang'] + 34),
                       (OFF['AnGang'], OFF['AnGang'] + 34), (OFF['BuGang'], OFF['BuGang'] + 34)]:
            assert set(A[lo:hi].tolist()) == set(range(lo, hi)), f"block {lo}:{hi} leaks for {q}"
        # Chi block untouched (identity)
        assert np.array_equal(A[OFF['Chi']:OFF['Chi'] + 63], np.arange(OFF['Chi'], OFF['Chi'] + 63))
        # winds (idx 27-30) fixed in tile map
        dt = _tile_perm(q); assert np.array_equal(dt[27:31], np.arange(27, 31))
    # identity perm changes nothing
    q0 = (0, 1, 2); assert np.array_equal(action_perm(q0), np.arange(235))
    # a transposition (swap J1<->J2) is involutive
    qt = (1, 0, 2); A = action_perm(qt); assert np.array_equal(A[A], np.arange(235))
    # obs round-trip on a transposition; winds/number rows untouched
    o = np.random.RandomState(0).randint(0, 2, (5, 38, 4, 9)).astype(np.int8)
    assert np.array_equal(dragon_obs(dragon_obs(o, qt), qt), o), "dragon_obs transposition not involutive"
    assert np.array_equal(dragon_obs(o, qt)[..., 3, :4], o[..., 3, :4]), "wind cols changed"
    assert np.array_equal(dragon_obs(o, qt)[..., :3, :], o[..., :3, :]), "number rows changed"
    # dragon cols actually swapped
    d = dragon_obs(o, qt)
    assert np.array_equal(d[..., 3, 4], o[..., 3, 5]) and np.array_equal(d[..., 3, 5], o[..., 3, 4])
    print("dragon_aug SELF-TESTS PASS: group + block + involution + obs round-trip OK (6x dragon aug ready)")
